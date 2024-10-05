import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from treys import Card, Evaluator, Deck
import pandas as pd
from collections import defaultdict

def simulate_poker_hands(num_trials=100000):
    evaluator = Evaluator()
    results_sb = defaultdict(lambda: [0, 0, defaultdict(int)])  # [wins, total games, hand_strengths] for small blind
    results_bb = defaultdict(lambda: [0, 0, defaultdict(int)])  # [wins, total games, hand_strengths] for big blind

    for _ in range(num_trials):
        deck = Deck()
        hand_sb = deck.draw(2)
        hand_bb = deck.draw(2)
        board = deck.draw(5)

        score_sb = evaluator.evaluate(board, hand_sb)
        score_bb = evaluator.evaluate(board, hand_bb)

        hand_sb_key = get_hand_key(hand_sb)
        hand_bb_key = get_hand_key(hand_bb)

        results_sb[hand_sb_key][1] += 1
        results_bb[hand_bb_key][1] += 1

        # Record hand strengths
        hand_strength_sb = evaluator.class_to_string(evaluator.get_rank_class(score_sb))
        hand_strength_bb = evaluator.class_to_string(evaluator.get_rank_class(score_bb))
        results_sb[hand_sb_key][2][hand_strength_sb] += 1
        results_bb[hand_bb_key][2][hand_strength_bb] += 1

        if score_sb < score_bb:
            results_sb[hand_sb_key][0] += 1
        elif score_bb < score_sb:
            results_bb[hand_bb_key][0] += 1
        else:
            results_sb[hand_sb_key][0] += 0.5
            results_bb[hand_bb_key][0] += 0.5

    return {
        'sb': {k: (v[0] / v[1], v[2]) for k, v in results_sb.items()},
        'bb': {k: (v[0] / v[1], v[2]) for k, v in results_bb.items()}
    }

def get_hand_key(hand):
    ranks = [Card.get_rank_int(card) for card in hand]
    suits = [Card.get_suit_int(card) for card in hand]
    ranks.sort(reverse=True)
    suited = 's' if suits[0] == suits[1] else 'o'
    return f"{Card.STR_RANKS[ranks[0]]}{Card.STR_RANKS[ranks[1]]}{suited}"

def create_hand_matrix():
    ranks = 'AKQJT98765432'
    return {r1 + r2 + s: 0 for r1 in ranks for r2 in ranks for s in ['s', 'o'] if r1 >= r2}

def plot_results(results):
    ranks = 'AKQJT98765432'
    matrix = [[0 for _ in range(13)] for _ in range(13)]
    suited_matrix = [['' for _ in range(13)] for _ in range(13)]

    max_winrate = max(max(v[0] for v in results['sb'].values()), max(v[0] for v in results['bb'].values()))

    for hand, (value, _) in results['sb'].items():
        i = ranks.index(hand[0])
        j = ranks.index(hand[1])
        suited = hand[2]
        if i == j:
            matrix[i][j] = value
            suited_matrix[i][j] = ''
        elif suited == 's':
            matrix[i][j] = value
            suited_matrix[i][j] = 's'
        else:
            matrix[j][i] = value
            suited_matrix[j][i] = 'o'

    for hand, (value, _) in results['bb'].items():
        i = ranks.index(hand[0])
        j = ranks.index(hand[1])
        suited = hand[2]
        if i == j:
            matrix[i][j] = (matrix[i][j] + value) / 2
        elif suited == 's':
            matrix[i][j] = (matrix[i][j] + value) / 2
        else:
            matrix[j][i] = (matrix[j][i] + value) / 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 13))

    # Plot heatmap with a custom colormap
    custom_cmap = sns.diverging_palette(10, 130, as_cmap=True)
    sns.heatmap(matrix, ax=ax1, cmap=custom_cmap, vmin=0, vmax=max_winrate, cbar=True, annot=False)

    for i in range(13):
        for j in range(13):
            if i == j:
                hand = ranks[i] + ranks[i]
                winrate = matrix[i][j]
                ax1.text(j + 0.5, i + 0.5, hand, ha='center', va='center', color='black', fontweight='bold')
                ax1.text(j + 0.5, i + 0.8, f'{winrate:.3f}', ha='center', va='center', color='black')
            elif i < j:
                hand = ranks[i] + ranks[j] + 's'
                winrate = matrix[i][j]
                ax1.text(j + 0.5, i + 0.3, hand, ha='center', va='center', color='black', fontweight='bold')
                ax1.text(j + 0.5, i + 0.7, f'{winrate:.3f}', ha='center', va='center', color='black')
            else:
                hand = ranks[j] + ranks[i] + 'o'
                winrate = matrix[i][j]
                ax1.text(j + 0.5, i + 0.3, hand, ha='center', va='center', color='black', fontweight='bold')
                ax1.text(j + 0.5, i + 0.7, f'{winrate:.3f}', ha='center', va='center', color='black')

    # Set title and labels for the first subplot
    ax1.set_title('Preflop Equity')
    ax1.set_xticklabels(ranks)
    ax1.set_yticklabels(ranks)

    # Create a function to update the hand strength distribution plot
    def update_hand_strength(event):
        if event.inaxes == ax1:
            i, j = int(event.ydata), int(event.xdata)
            if i == j:
                hand = ranks[i] + ranks[i]
            elif i < j:
                hand = ranks[i] + ranks[j] + 's'
            else:
                hand = ranks[j] + ranks[i] + 'o'

            ax2.clear()
            hand_strengths = results['sb'][hand][1]
            total = sum(hand_strengths.values())
            percentages = {k: v / total * 100 for k, v in hand_strengths.items()}
            
            ax2.bar(percentages.keys(), percentages.values())
            ax2.set_title(f'Hand Strength Distribution for {hand}')
            ax2.set_xlabel('Hand Strength')
            ax2.set_ylabel('Percentage')
            ax2.set_ylim(0, 100)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            fig.canvas.draw()

    # Connect the event to the plot
    fig.canvas.mpl_connect('button_press_event', update_hand_strength)

    # Set title for the second subplot
    ax2.set_title('Click on a hand to see its strength distribution')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    results = simulate_poker_hands()
    plot_results(results)
