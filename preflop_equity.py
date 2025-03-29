import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from treys import Card, Evaluator
from collections import defaultdict
import itertools
import random
from tqdm import tqdm

def calculate_poker_equity():
    """Calculate preflop equity for all possible starting hands with optimized performance"""
    evaluator = Evaluator()
    ranks = 'AKQJT98765432'
    
    # Initialize results structure
    print("Initializing hand combinations...")
    results = {}
    for position in ['sb', 'bb']:
        results[position] = {}
        for r1 in ranks:
            for r2 in ranks:
                if r1 >= r2:  # Only consider unique hands
                    for suited in ['s', 'o']:
                        # Skip invalid combinations (pairs can't be suited)
                        if r1 == r2 and suited == 's':
                            continue
                        # Skip offsuit when same rank (redundant with pairs)
                        if r1 == r2 and suited == 'o':
                            continue
                        hand_key = f"{r1}{r2}{suited}" if r1 != r2 else f"{r1}{r2}"
                        results[position][hand_key] = [0, 0, defaultdict(int)]  # [wins, total, hand_strengths]
    
    # Calculate equity for each possible hand vs hand matchup
    all_hands = list(results['sb'].keys())
    total_matchups = sum(len(all_hands) - i for i in range(len(all_hands)))
    print(f"Calculating equity for {total_matchups} possible hand matchups...")
    
    # Create progress bar for the outer loop
    progress_bar = tqdm(total=total_matchups, desc="Calculating hand matchups", unit="matchup")
    
    # For each possible hand matchup
    for i, hand1_key in enumerate(all_hands):
        for hand2_key in all_hands[i:]:  # Only calculate each matchup once
            # Skip impossible matchups (same cards)
            if hand1_key == hand2_key and len(hand1_key) == 2:  # Pairs
                progress_bar.update(1)  # Update progress bar even for skipped matchups
                continue
            if hand1_key[:2] == hand2_key[:2] and hand1_key[2] != hand2_key[2]:  # Same ranks different suits
                progress_bar.update(1)  # Update progress bar even for skipped matchups
                continue
                
            # Convert hand keys to actual cards
            hand1_cards = convert_key_to_cards(hand1_key)
            hand2_cards = convert_key_to_cards(hand2_key)
            
            # Skip impossible matchups (shared cards)
            if any(card in hand2_cards for card in hand1_cards):
                progress_bar.update(1)  # Update progress bar even for skipped matchups
                continue
                
            # Calculate equity
            equity1, equity2, hand_strengths1, hand_strengths2 = calculate_equity(hand1_cards, hand2_cards, evaluator)
            
            # Update results
            matchup_count = 1
            results['sb'][hand1_key][0] += equity1 * matchup_count
            results['bb'][hand2_key][0] += equity2 * matchup_count
            results['sb'][hand1_key][1] += matchup_count
            results['bb'][hand2_key][1] += matchup_count
            
            # Update progress bar
            progress_bar.update(1)
            
            # Update hand strength distributions
            for hand_strength, count in hand_strengths1.items():
                results['sb'][hand1_key][2][hand_strength] += count
            for hand_strength, count in hand_strengths2.items():
                results['bb'][hand2_key][2][hand_strength] += count
            
            # Also update for the reverse matchup (BB vs SB)
            if hand1_key != hand2_key:  # Only if hands are different
                results['bb'][hand1_key][0] += equity1 * matchup_count
                results['sb'][hand2_key][0] += equity2 * matchup_count
                results['bb'][hand1_key][1] += matchup_count
                results['sb'][hand2_key][1] += matchup_count
                
                # Update hand strength distributions for reverse matchup
                for hand_strength, count in hand_strengths1.items():
                    results['bb'][hand1_key][2][hand_strength] += count
                for hand_strength, count in hand_strengths2.items():
                    results['sb'][hand2_key][2][hand_strength] += count
    
    # Close the progress bar
    progress_bar.close()
    
    # Calculate final equity percentages
    print("Finalizing equity calculations...")
    for position in ['sb', 'bb']:
        for hand_key in results[position]:
            if results[position][hand_key][1] > 0:  # Avoid division by zero
                win_rate = results[position][hand_key][0] / results[position][hand_key][1]
                results[position][hand_key] = (win_rate, results[position][hand_key][2])
            else:
                results[position][hand_key] = (0, defaultdict(int))
    
    return results

def convert_key_to_cards(hand_key):
    """Convert a hand key (e.g., 'AKs', 'TT') to actual card objects"""
    if len(hand_key) == 2:  # Pair
        rank = hand_key[0]
        # Create a pair with different suits
        card1 = Card.new(rank + 'h')
        card2 = Card.new(rank + 'd')
    else:  # Non-pair
        rank1, rank2 = hand_key[0], hand_key[1]
        suited = hand_key[2] == 's'
        
        if suited:
            card1 = Card.new(rank1 + 'h')
            card2 = Card.new(rank2 + 'h')
        else:  # Offsuit
            card1 = Card.new(rank1 + 'h')
            card2 = Card.new(rank2 + 'd')
    
    return [card1, card2]

def calculate_equity(hand1, hand2, evaluator):
    """Calculate equity between two specific hands using optimized Monte Carlo simulation"""
    # Create a deck without the cards in hand1 and hand2
    all_cards = [Card.new(rank + suit) for rank in Card.STR_RANKS for suit in 'hdcs']
    available_cards = [card for card in all_cards if card not in hand1 and card not in hand2]
    
    # Reduced sample size for faster computation
    sample_size = 500  # Reduced from 1000 for better performance
    
    # Instead of generating all combinations, directly sample random boards
    # This is much faster than generating all combinations and then sampling
    boards_to_evaluate = []
    for _ in range(sample_size):
        boards_to_evaluate.append(random.sample(available_cards, 5))
    
    wins1, wins2, ties = 0, 0, 0
    hand_strengths1 = defaultdict(int)
    hand_strengths2 = defaultdict(int)
    
    # Evaluate boards without nested progress bar for cleaner output
    for board in boards_to_evaluate:
        score1 = evaluator.evaluate(board, hand1)
        score2 = evaluator.evaluate(board, hand2)
        
        # Record hand strengths
        hand_strength1 = evaluator.class_to_string(evaluator.get_rank_class(score1))
        hand_strength2 = evaluator.class_to_string(evaluator.get_rank_class(score2))
        hand_strengths1[hand_strength1] += 1
        hand_strengths2[hand_strength2] += 1
        
        if score1 < score2:  # Lower score is better in treys
            wins1 += 1
        elif score2 < score1:
            wins2 += 1
        else:
            ties += 1
    
    total = len(boards_to_evaluate)
    equity1 = (wins1 + ties/2) / total
    equity2 = (wins2 + ties/2) / total
    
    return equity1, equity2, hand_strengths1, hand_strengths2

def plot_results(results):
    ranks = 'AKQJT98765432'
    matrix = [[0 for _ in range(13)] for _ in range(13)]
    suited_matrix = [['' for _ in range(13)] for _ in range(13)]
    
    max_winrate = max(max(v[0] for v in results['sb'].values()), max(v[0] for v in results['bb'].values()))

    # Populate matrix with results
    for hand, (value, _) in results['sb'].items():
        i = ranks.index(hand[0])
        j = ranks.index(hand[1])
        if len(hand) == 2:  # Pair
            matrix[i][j] = value
            suited_matrix[i][j] = ''
        elif hand[2] == 's':  # Suited
            matrix[i][j] = value
            suited_matrix[i][j] = 's'
        else:  # Offsuit
            matrix[j][i] = value
            suited_matrix[j][i] = 'o'

    for hand, (value, _) in results['bb'].items():
        i = ranks.index(hand[0])
        j = ranks.index(hand[1])
        if len(hand) == 2:  # Pair
            matrix[i][j] = (matrix[i][j] + value) / 2
        elif hand[2] == 's':  # Suited
            matrix[i][j] = (matrix[i][j] + value) / 2
        else:  # Offsuit
            matrix[j][i] = (matrix[j][i] + value) / 2

    # Create a figure with a dark background for a modern look
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 12), facecolor='#1e1e1e')
    
    # Create a grid layout for better organization
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[4, 1])
    
    # Main heatmap in the left area
    ax_heatmap = fig.add_subplot(gs[0, 0])
    
    # Distribution plot in the top-right
    ax_dist = fig.add_subplot(gs[0, 1:])
    
    # Additional stats in the bottom row
    ax_stats = fig.add_subplot(gs[1, :])
    
    # Create a custom colormap with poker-themed colors (red to green gradient)
    custom_cmap = plt.colormaps['RdYlGn']
    
    # Plot the heatmap with enhanced styling
    heatmap = sns.heatmap(matrix, ax=ax_heatmap, cmap=custom_cmap, vmin=0, vmax=max_winrate, 
                        cbar=True, annot=False, linewidths=0.5, linecolor='#333333')
    
    # Add a stylish title
    fig.suptitle('Poker Preflop Equity Analysis', fontsize=24, color='white', fontweight='bold', y=0.98)
    
    # Customize the colorbar
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Win Rate', fontsize=14, color='white')
    cbar.ax.tick_params(colors='white')
    
    # Add hand labels with improved styling
    for i in range(13):
        for j in range(13):
            # Determine text color based on cell value for better contrast
            cell_value = matrix[i][j]
            text_color = 'white' if cell_value < max_winrate * 0.6 else 'black'
            
            if i == j:  # Pairs
                hand = ranks[i] + ranks[i]
                winrate = matrix[i][j]
                # Create a more visually appealing text layout
                ax_heatmap.text(j + 0.5, i + 0.5, hand, ha='center', va='center', 
                            color=text_color, fontweight='bold', fontsize=12)
                ax_heatmap.text(j + 0.5, i + 0.75, f'{winrate:.3f}', ha='center', va='center', 
                            color=text_color, fontsize=10)
            elif i < j:  # Suited hands
                hand = ranks[i] + ranks[j] + 's'
                winrate = matrix[i][j]
                ax_heatmap.text(j + 0.5, i + 0.35, hand, ha='center', va='center', 
                            color=text_color, fontweight='bold', fontsize=12)
                ax_heatmap.text(j + 0.5, i + 0.65, f'{winrate:.3f}', ha='center', va='center', 
                            color=text_color, fontsize=10)
            else:  # Offsuit hands
                hand = ranks[j] + ranks[i] + 'o'
                winrate = matrix[i][j]
                ax_heatmap.text(j + 0.5, i + 0.35, hand, ha='center', va='center', 
                            color=text_color, fontweight='bold', fontsize=12)
                ax_heatmap.text(j + 0.5, i + 0.65, f'{winrate:.3f}', ha='center', va='center', 
                            color=text_color, fontsize=10)
    
    # Set axis labels with improved styling
    ax_heatmap.set_title('Preflop Hand Equity Matrix', fontsize=18, color='white', pad=20)
    ax_heatmap.set_xticklabels(ranks, fontsize=12, color='white')
    ax_heatmap.set_yticklabels(ranks, fontsize=12, color='white')
    ax_heatmap.tick_params(colors='white')
    
    # Add hand category labels
    ax_heatmap.text(6.5, -1.5, 'Card Ranks', ha='center', fontsize=14, color='white')
    ax_heatmap.text(-1.5, 6.5, 'Card Ranks', ha='center', fontsize=14, color='white', rotation=90)
    
    # Initialize the distribution plot with a placeholder
    ax_dist.set_title('Hand Strength Distribution', fontsize=16, color='white')
    ax_dist.set_xlabel('Hand Strength', fontsize=12, color='white')
    ax_dist.set_ylabel('Percentage (%)', fontsize=12, color='white')
    ax_dist.tick_params(colors='white')
    ax_dist.grid(True, linestyle='--', alpha=0.3)
    ax_dist.text(0.5, 0.5, 'Click on a hand in the matrix\nto see its distribution', 
                ha='center', va='center', fontsize=14, color='white', transform=ax_dist.transAxes)
    
    # Initialize the stats area with overall statistics
    ax_stats.axis('off')
    ax_stats.text(0.01, 0.7, 'Overall Statistics:', fontsize=16, color='white', fontweight='bold')
    ax_stats.text(0.01, 0.4, f'Average Win Rate: {np.mean([v[0] for v in results["sb"].values()]):.3f}', 
                fontsize=14, color='white')
    ax_stats.text(0.3, 0.4, f'Highest Win Rate: {max_winrate:.3f}', fontsize=14, color='white')
    
    # For total hands, we'll use the sample size from calculate_equity
    total_hands = 500  # This matches the optimized sample size
    ax_stats.text(0.6, 0.4, f'Total Hands Analyzed: {total_hands}', 
                 fontsize=14, color='white')
    
    # Create a function to update the hand strength distribution plot with improved visualization
    def update_hand_strength(event):
        if event.inaxes == ax_heatmap:
            try:
                i, j = int(event.ydata), int(event.xdata)
                if 0 <= i < 13 and 0 <= j < 13:  # Ensure valid indices
                    # Clear previous highlights
                    for patch in ax_heatmap.patches:
                        if isinstance(patch, plt.Rectangle) and patch.get_edgecolor() != 'none':
                            patch.remove()
                    
                    
                    # Determine hand type and key
                    if i == j:
                        hand = ranks[i] + ranks[i]
                        hand_type = 'Pocket Pair'
                    elif i < j:
                        hand = ranks[i] + ranks[j] + 's'
                        hand_type = 'Suited'
                    else:
                        hand = ranks[j] + ranks[i] + 'o'
                        hand_type = 'Offsuit'

                    # Clear previous plots
                    ax_dist.clear()
                    
                    # Get hand strength data
                    if hand in results['sb']:
                        hand_strengths = results['sb'][hand][1]
                        total = sum(hand_strengths.values())
                        
                        if total > 0:  # Ensure we have data
                            # Sort hand strengths by poker hand ranking
                            hand_ranking = ['High Card', 'Pair', 'Two Pair', 'Three of a Kind', 
                                            'Straight', 'Flush', 'Full House', 'Four of a Kind', 'Straight Flush']
                            sorted_strengths = {k: hand_strengths.get(k, 0) for k in hand_ranking if k in hand_strengths}
                            percentages = {k: v / total * 100 for k, v in sorted_strengths.items()}
                            
                            # Create a more visually appealing bar chart
                            colors = plt.cm.viridis(np.linspace(0, 0.8, len(percentages)))
                            bars = ax_dist.bar(list(percentages.keys()), list(percentages.values()), color=colors, width=0.7)
                            
                            # Add percentage labels on top of bars
                            for bar in bars:
                                height = bar.get_height()
                                if height > 5:  # Only show label if percentage is significant
                                    ax_dist.text(bar.get_x() + bar.get_width()/2., height + 1,
                                                f'{height:.1f}%', ha='center', va='bottom', color='white', fontsize=10)
                            
                            # Update title and styling
                            ax_dist.set_title(f'Hand Strength Distribution for {hand} ({hand_type})', 
                                            fontsize=16, color='white')
                            ax_dist.set_xlabel('Hand Strength', fontsize=12, color='white')
                            ax_dist.set_ylabel('Percentage (%)', fontsize=12, color='white')
                            ax_dist.set_ylim(0, max(percentages.values()) * 1.2 if percentages else 100)
                            ax_dist.tick_params(colors='white')
                            ax_dist.grid(True, linestyle='--', alpha=0.3)
                            plt.setp(ax_dist.get_xticklabels(), rotation=45, ha='right', fontsize=10)
                            
                            # Update stats area with selected hand information
                            ax_stats.clear()
                            ax_stats.axis('off')
                            ax_stats.text(0.01, 0.7, f'Statistics for {hand}:', fontsize=16, color='white', fontweight='bold')
                            ax_stats.text(0.01, 0.4, f'Win Rate: {matrix[i][j]:.3f}', fontsize=14, color='white')
                            ax_stats.text(0.3, 0.4, f'Hand Type: {hand_type}', fontsize=14, color='white')
                            
                            if sorted_strengths:
                                most_common = max(sorted_strengths.items(), key=lambda x: x[1])[0]
                                ax_stats.text(0.6, 0.4, f'Most Common Outcome: {most_common}', 
                                            fontsize=14, color='white')
                            
                            # Add highlight rectangle to the selected cell
                            rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='white', linewidth=3)
                            ax_heatmap.add_patch(rect)
                    
                    fig.canvas.draw_idle()  # More efficient than full redraw
            except Exception as e:
                print(f"Error in update_hand_strength: {e}")

    # Connect the event to the plot
    fig.canvas.mpl_connect('button_press_event', update_hand_strength)
    
    # Add a legend for hand types
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=custom_cmap(0.8), markersize=15, label='High Win Rate'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=custom_cmap(0.5), markersize=15, label='Medium Win Rate'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=custom_cmap(0.2), markersize=15, label='Low Win Rate')
    ]
    ax_heatmap.legend(handles=legend_elements, loc='upper right', framealpha=0.7, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.05, wspace=0.2, hspace=0.3)
    plt.show()

if __name__ == '__main__':
    print("Starting preflop equity calculator (optimized version)...")
    print("This will take a few minutes. Progress will be shown below:")
    results = calculate_poker_equity()
    print("Calculation complete. Generating visualization...")
    plot_results(results)
    print("Visualization complete. Click on any hand in the matrix to see detailed statistics.")
