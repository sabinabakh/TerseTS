import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import os

def load_results(csv_path):
    """Load benchmark results from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        # Filter only successful results
        df = df[df['success']].copy()
        print(f"Loaded {len(df)} successful benchmark results")
        return df
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        return None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def analyze_best_compression(df):
    """Analyze which method has the best timestamp compression ratio for each combination."""
    # Group by dataset, compression_method, and error_bound to find best timestamp encoding method
    best_compression = {}
    
    # Group by unique combination of dataset, compression method, and error bound
    grouped = df.groupby(['dataset', 'compression_method', 'error_bound'])
    
    for (dataset, comp_method, error_bound), group in grouped:
        # Find the method with the highest timestamps_encoding_ratio
        best_row = group.loc[group['timestamps_encoding_ratio'].idxmax()]
        best_method = best_row['timestamps_encoding_method']
        
        key = f"{dataset}_{comp_method}_{error_bound}"
        best_compression[key] = best_method
    
    return best_compression

def analyze_best_time(df):
    """Analyze which method has the best (lowest) timestamp encoding time for each combination."""
    # Group by dataset, compression_method, and error_bound to find best timestamp encoding method
    best_time = {}
    
    # Group by unique combination of dataset, compression method, and error bound
    grouped = df.groupby(['dataset', 'compression_method', 'error_bound'])
    
    for (dataset, comp_method, error_bound), group in grouped:
        # Find the method with the lowest timestamps_encoding_time_ns
        best_row = group.loc[group['timestamps_encoding_time_ns'].idxmin()]
        best_method = best_row['timestamps_encoding_method']
        
        key = f"{dataset}_{comp_method}_{error_bound}"
        best_time[key] = best_method
    
    return best_time

def analyze_best_compression_size(df):
    """Analyze which method achieves the smallest compressed size for timestamps."""
    # Group by dataset, compression_method, and error_bound to find best timestamp encoding method
    best_size = {}
    
    # Group by unique combination of dataset, compression method, and error bound
    grouped = df.groupby(['dataset', 'compression_method', 'error_bound'])
    
    for (dataset, comp_method, error_bound), group in grouped:
        # Find the method with the smallest timestamps_encoded_bytes
        best_row = group.loc[group['timestamps_encoded_bytes'].idxmin()]
        best_method = best_row['timestamps_encoding_method']
        
        key = f"{dataset}_{comp_method}_{error_bound}"
        best_size[key] = best_method
    
    return best_size

def calculate_percentages(best_methods):
    """Calculate percentage for each method."""
    method_counts = defaultdict(int)
    total_cases = len(best_methods)
    
    for method in best_methods.values():
        method_counts[method] += 1
    
    # Convert to percentages
    percentages = {}
    print(method_counts)
    for method, count in method_counts.items():
        percentages[method] = (count / total_cases) * 100
    
    return percentages, total_cases

def analyze_best_combinations(df):
    """Analyze which combination of functional + integer methods performs best overall."""
    # Group by dataset and error_bound to find the single best combination across ALL methods
    best_combinations = {}
    
    # Group by unique combination of dataset and error bound  
    grouped = df.groupby(['dataset', 'error_bound'])
    
    for (dataset, error_bound), group in grouped:
        # Find the single combination with the highest total_compression_ratio across ALL methods
        best_row = group.loc[group['total_compression_ratio'].idxmax()]
        combo = f"{best_row['compression_method']} + {best_row['timestamps_encoding_method']}"
        
        key = f"{dataset}_{error_bound}"
        best_combinations[key] = combo
    
    return best_combinations

def analyze_best_time_combinations(df):
    """Analyze which combination of functional + integer methods achieves the fastest overall time."""
    # Group by dataset and error_bound to find the single fastest combination across ALL methods
    best_time_combinations = {}
    
    # Group by unique combination of dataset and error bound  
    grouped = df.groupby(['dataset', 'error_bound'])
    
    for (dataset, error_bound), group in grouped:
        # Find the single combination with the lowest total_time_ns across ALL methods
        best_row = group.loc[group['total_time_ns'].idxmin()]
        combo = f"{best_row['compression_method']} + {best_row['timestamps_encoding_method']}"
        
        key = f"{dataset}_{error_bound}"
        best_time_combinations[key] = combo
    
    return best_time_combinations

def analyze_best_balanced_combinations(df):
    """Analyze which combination performs best based on both compression ratio and time."""
    # Group by dataset and error_bound to find the best balanced combination
    best_balanced_combinations = {}
    
    # Group by unique combination of dataset and error bound  
    grouped = df.groupby(['dataset', 'error_bound'])
    
    for (dataset, error_bound), group in grouped:
        # Normalize both metrics to 0-1 scale within this group
        group = group.copy()
        
        # Higher compression ratio is better (normalize to 0-1, higher = better)
        min_ratio = group['total_compression_ratio'].min()
        max_ratio = group['total_compression_ratio'].max()
        if max_ratio > min_ratio:
            group['norm_ratio'] = (group['total_compression_ratio'] - min_ratio) / (max_ratio - min_ratio)
        else:
            group['norm_ratio'] = 1.0  # All same, assign max score
        
        # Lower time is better (normalize to 0-1, higher = better)
        min_time = group['total_time_ns'].min()
        max_time = group['total_time_ns'].max()
        if max_time > min_time:
            group['norm_time'] = 1.0 - (group['total_time_ns'] - min_time) / (max_time - min_time)
        else:
            group['norm_time'] = 1.0  # All same, assign max score
        
        # Combined score: 60% compression ratio weight, 40% time weight
        group['balanced_score'] = 0.6 * group['norm_ratio'] + 0.4 * group['norm_time']
        
        # Find the combination with the highest balanced score
        best_row = group.loc[group['balanced_score'].idxmax()]
        combo = f"{best_row['compression_method']} + {best_row['timestamps_encoding_method']}"
        
        key = f"{dataset}_{error_bound}"
        best_balanced_combinations[key] = combo
    
    return best_balanced_combinations

def analyze_average_total_compression(df):
    """Analyze average total compression ratio for each integer encoding method."""
    # Group by integer encoding method and calculate average total compression ratio
    method_averages = {}
    
    for method in df['timestamps_encoding_method'].unique():
        method_df = df[df['timestamps_encoding_method'] == method]
        avg_compression = method_df['total_compression_ratio'].mean()
        method_averages[method] = avg_compression
    
    return method_averages

def analyze_average_time(df):
    """Analyze average total time for each integer encoding method."""
    # Group by integer encoding method and calculate average total time
    method_averages = {}
    
    for method in df['timestamps_encoding_method'].unique():
        method_df = df[df['timestamps_encoding_method'] == method]
        avg_time_seconds = method_df['total_time_ns'].mean() / 1e9  # Convert to seconds
        method_averages[method] = avg_time_seconds
    
    return method_averages

def analyze_average_timestamps_compression(df):
    """Analyze average timestamps compression ratio for each integer encoding method."""
    # Group by integer encoding method and calculate average timestamps compression ratio
    method_averages = {}
    
    for method in df['timestamps_encoding_method'].unique():
        method_df = df[df['timestamps_encoding_method'] == method]
        avg_timestamps_ratio = method_df['timestamps_encoding_ratio'].mean()
        method_averages[method] = avg_timestamps_ratio
    
    return method_averages

def analyze_average_timestamps_time(df):
    """Analyze average timestamps encoding time for each integer encoding method."""
    # Group by integer encoding method and calculate average timestamps encoding time
    method_averages = {}
    
    for method in df['timestamps_encoding_method'].unique():
        method_df = df[df['timestamps_encoding_method'] == method]
        avg_timestamps_time_seconds = method_df['timestamps_encoding_time_ns'].mean() / 1e9  # Convert to seconds
        method_averages[method] = avg_timestamps_time_seconds
    
    return method_averages

def analyze_dataset_performance(df):
    """Analyze how different datasets respond to different method combinations."""
    dataset_winners = {}
    
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        dataset_winners[dataset] = {}
        
        # Best overall compression ratio for this dataset
        best_ratio_row = dataset_df.loc[dataset_df['total_compression_ratio'].idxmax()]
        dataset_winners[dataset]['best_compression'] = f"{best_ratio_row['compression_method']} + {best_ratio_row['timestamps_encoding_method']}"
        
        # Best time for this dataset
        best_time_row = dataset_df.loc[dataset_df['total_time_ns'].idxmin()]
        dataset_winners[dataset]['best_time'] = f"{best_time_row['compression_method']} + {best_time_row['timestamps_encoding_method']}"
        
        # Most consistent performer (appears most often as winner across error bounds)
        eb_grouped = dataset_df.groupby(['error_bound'])
        best_combos = {}
        for error_bound, group in eb_grouped:
            best_row = group.loc[group['total_compression_ratio'].idxmax()]
            combo = f"{best_row['compression_method']} + {best_row['timestamps_encoding_method']}"
            best_combos[error_bound] = combo
        
        combo_counts = defaultdict(int)
        for combo in best_combos.values():
            combo_counts[combo] += 1
        
        if combo_counts:
            most_consistent = max(combo_counts.items(), key=lambda x: x[1])
            dataset_winners[dataset]['most_consistent'] = most_consistent[0]
            dataset_winners[dataset]['consistency_score'] = most_consistent[1]
    
    return dataset_winners

def analyze_error_bound_impact(df):
    """Analyze how error bounds affect method selection and performance."""
    error_bound_analysis = {}
    
    for error_bound in df['error_bound'].unique():
        eb_df = df[df['error_bound'] == error_bound]
        error_bound_analysis[error_bound] = {}
        
        # Average performance metrics for this error bound
        error_bound_analysis[error_bound]['avg_compression_ratio'] = eb_df['total_compression_ratio'].mean()
        error_bound_analysis[error_bound]['avg_time_seconds'] = eb_df['total_time_ns'].mean() / 1e9
        error_bound_analysis[error_bound]['avg_size_mb'] = eb_df['total_compressed_bytes'].mean() / (1024 * 1024)
        
        # Most popular method combination at this error bound
        dataset_grouped = eb_df.groupby(['dataset'])
        best_combos = {}
        for dataset, group in dataset_grouped:
            best_row = group.loc[group['total_compression_ratio'].idxmax()]
            combo = f"{best_row['compression_method']} + {best_row['timestamps_encoding_method']}"
            best_combos[dataset] = combo
        
        combo_counts = defaultdict(int)
        for combo in best_combos.values():
            combo_counts[combo] += 1
        
        if combo_counts:
            most_popular = max(combo_counts.items(), key=lambda x: x[1])
            error_bound_analysis[error_bound]['most_popular_combo'] = most_popular[0]
    
    return error_bound_analysis

def create_pie_chart(percentages, title, filename):
    """Create a clean, simple pie chart showing the percentages."""
    methods = list(percentages.keys())
    values = list(percentages.values())
    
    # Ensure diagrams directory exists
    os.makedirs('analyse_results_diagrams', exist_ok=True)
    
    # Create figure with clean styling
    plt.figure(figsize=(10, 8))
    
    # Simple color palette
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF']
    
    # Create simple pie chart
    wedges, texts, autotexts = plt.pie(values, 
                                       labels=methods, 
                                       autopct='%1.1f%%',
                                       startangle=90, 
                                       colors=colors[:len(methods)])
    
    # Simple text formatting
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    for text in texts:
        text.set_fontsize(10)
        # Wrap long method names
        method_name = text.get_text()
        if len(method_name) > 20:
            if 'Encoding' in method_name:
                wrapped_name = method_name.replace('Encoding', '\nEncoding')
                text.set_text(wrapped_name)
    
    # Simple title
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Equal aspect ratio for perfect circle
    plt.axis('equal')
    
    # Save with standard quality
    plt.tight_layout()
    plt.savefig(f'analyse_results_diagrams/{filename}', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Chart saved as: analyse_results_diagrams/{filename}")

def create_combination_chart(percentages, title, filename):
    """Create a horizontal bar chart for combination analysis."""
    # Sort combinations by percentage
    sorted_combos = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
    
    # Take top 10 combinations to avoid overcrowding
    top_combos = sorted_combos[:10]
    
    combinations = [combo[0] for combo in top_combos]
    values = [combo[1] for combo in top_combos]
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Create horizontal bar chart
    bars = plt.barh(range(len(combinations)), values, color='#4CAF50')
    
    # Customize the chart
    plt.xlabel('Percentage (%)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Set y-axis labels with proper spacing
    plt.yticks(range(len(combinations)), combinations, fontsize=10)
    
    # Add percentage labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}%', ha='left', va='center', fontweight='bold')
    
    # Add grid for better readability
    plt.grid(axis='x', alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save chart
    plt.savefig(f'analyse_results_diagrams/{filename}', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Chart saved as: analyse_results_diagrams/{filename}")

def create_dataset_comparison_chart(dataset_winners):
    """Create a comparison chart showing how different datasets perform with different methods."""
    datasets = list(dataset_winners.keys())
    categories = ['Best Compression', 'Best Time', 'Most Consistent']
    
    fig, axes = plt.subplots(1, len(categories), figsize=(18, 6))
    
    for i, category in enumerate(categories):
        category_key = category.lower().replace(' ', '_')
        
        # Get data for this category
        if category_key == 'most_consistent':
            data = {dataset: dataset_winners[dataset].get('most_consistent', 'N/A') 
                   for dataset in datasets}
        else:
            data = {dataset: dataset_winners[dataset].get(category_key, 'N/A') 
                   for dataset in datasets}
        
        # Count occurrences of each method combination
        method_counts = defaultdict(int)
        for method in data.values():
            if method != 'N/A':
                method_counts[method] += 1
        
        # Create pie chart for this category
        if method_counts:
            methods = list(method_counts.keys())
            values = list(method_counts.values())
            colors = plt.cm.Set3(range(len(methods)))
            
            axes[i].pie(values, labels=[m.replace(' + ', '\n+ ') for m in methods], 
                       autopct='%1.0f%%', startangle=90, colors=colors)
            axes[i].set_title(f"{category}\nby Dataset", fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('analyse_results_diagrams/dataset_performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Chart saved as: analyse_results_diagrams/dataset_performance_comparison.png")

def create_error_bound_analysis_chart(error_bound_analysis):
    """Create charts showing how error bounds affect performance metrics."""
    error_bounds = sorted(error_bound_analysis.keys())
    
    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Compression ratio trend
    ratios = [error_bound_analysis[eb]['avg_compression_ratio'] for eb in error_bounds]
    axes[0, 0].plot(error_bounds, ratios, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_title('Average Compression Ratio vs Error Bound', fontweight='bold')
    axes[0, 0].set_xlabel('Error Bound')
    axes[0, 0].set_ylabel('Avg Compression Ratio')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Time trend
    times = [error_bound_analysis[eb]['avg_time_seconds'] for eb in error_bounds]
    axes[0, 1].plot(error_bounds, times, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_title('Average Processing Time vs Error Bound', fontweight='bold')
    axes[0, 1].set_xlabel('Error Bound')
    axes[0, 1].set_ylabel('Avg Time (seconds)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Size trend
    sizes = [error_bound_analysis[eb]['avg_size_mb'] for eb in error_bounds]
    axes[1, 0].plot(error_bounds, sizes, 'go-', linewidth=2, markersize=8)
    axes[1, 0].set_title('Average Compressed Size vs Error Bound', fontweight='bold')
    axes[1, 0].set_xlabel('Error Bound')
    axes[1, 0].set_ylabel('Avg Size (MB)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Most popular combinations by error bound
    popular_combos = [error_bound_analysis[eb].get('most_popular_combo', 'N/A') 
                     for eb in error_bounds]
    
    # Create a simple text display for popular combinations
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Most Popular Combinations by Error Bound', fontweight='bold')
    
    y_pos = 0.8
    for eb, combo in zip(error_bounds, popular_combos):
        axes[1, 1].text(0.1, y_pos, f"Error Bound {eb}:", fontweight='bold', fontsize=11)
        axes[1, 1].text(0.1, y_pos-0.1, combo.replace(' + ', ' +\n'), fontsize=10)
        y_pos -= 0.25
    
    plt.tight_layout()
    plt.savefig('analyse_results_diagrams/error_bound_impact_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Chart saved as: analyse_results_diagrams/error_bound_impact_analysis.png")

def create_average_compression_chart(method_averages, title, filename):
    """Create a bar chart showing average compression ratios for each integer encoding method."""
    # Sort methods by average compression ratio (descending)
    sorted_methods = sorted(method_averages.items(), key=lambda x: x[1], reverse=True)
    
    methods = [method[0] for method in sorted_methods]
    values = [method[1] for method in sorted_methods]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create bar chart
    bars = plt.bar(range(len(methods)), values, color='#2E86AB', alpha=0.8)
    
    # Customize the chart
    plt.xlabel('Integer Encoding Method', fontsize=12)
    plt.ylabel('Average Total Compression Ratio', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis labels with rotation for readability
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right', fontsize=10)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save chart
    plt.savefig(f'analyse_results_diagrams/{filename}', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Chart saved as: analyse_results_diagrams/{filename}")

def create_average_time_chart(method_averages, title, filename):
    """Create a bar chart showing average processing times for each integer encoding method."""
    # Sort methods by average time (ascending - lower is better)
    sorted_methods = sorted(method_averages.items(), key=lambda x: x[1])
    
    methods = [method[0] for method in sorted_methods]
    values = [method[1] for method in sorted_methods]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create bar chart with color gradient (green for faster, red for slower)
    colors = plt.cm.RdYlGn_r([(v - min(values)) / (max(values) - min(values)) for v in values])
    bars = plt.bar(range(len(methods)), values, color=colors, alpha=0.8)
    
    # Customize the chart
    plt.xlabel('Integer Encoding Method', fontsize=12)
    plt.ylabel('Average Total Processing Time (seconds)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis labels with rotation for readability
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right', fontsize=10)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                f'{value:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save chart
    plt.savefig(f'analyse_results_diagrams/{filename}', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Chart saved as: analyse_results_diagrams/{filename}")

def create_timestamps_compression_chart(method_averages, title, filename):
    """Create a bar chart showing average timestamps compression ratios for each integer encoding method."""
    # Sort methods by average compression ratio (descending - higher is better)
    sorted_methods = sorted(method_averages.items(), key=lambda x: x[1], reverse=True)
    
    methods = [method[0] for method in sorted_methods]
    values = [method[1] for method in sorted_methods]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create bar chart with color gradient (green for better, red for worse)
    colors = plt.cm.RdYlGn([(v - min(values)) / (max(values) - min(values)) for v in values])
    bars = plt.bar(range(len(methods)), values, color=colors, alpha=0.8)
    
    # Customize the chart
    plt.xlabel('Integer Encoding Method', fontsize=12)
    plt.ylabel('Average Timestamps Compression Ratio', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis labels with rotation for readability
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right', fontsize=10)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save chart
    plt.savefig(f'analyse_results_diagrams/{filename}', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Chart saved as: analyse_results_diagrams/{filename}")

def create_timestamps_time_chart(method_averages, title, filename):
    """Create a bar chart showing average timestamps encoding times for each integer encoding method."""
    # Sort methods by average time (ascending - lower is better)
    sorted_methods = sorted(method_averages.items(), key=lambda x: x[1])
    
    methods = [method[0] for method in sorted_methods]
    values = [method[1] for method in sorted_methods]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create bar chart with color gradient (green for faster, red for slower)
    colors = plt.cm.RdYlGn_r([(v - min(values)) / (max(values) - min(values)) for v in values])
    bars = plt.bar(range(len(methods)), values, color=colors, alpha=0.8)
    
    # Customize the chart
    plt.xlabel('Integer Encoding Method', fontsize=12)
    plt.ylabel('Average Timestamps Encoding Time (seconds)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis labels with rotation for readability
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right', fontsize=10)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                f'{value:.4f}s', ha='center', va='bottom', fontweight='bold')
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save chart
    plt.savefig(f'analyse_results_diagrams/{filename}', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Chart saved as: analyse_results_diagrams/{filename}")

def print_detailed_stats(percentages, title):
    """Print detailed statistics with simple formatting."""
    print(f"\n{title}")
    print("-" * len(title))
    
    # Sort by percentage (descending)
    sorted_methods = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
    
    for i, (method, percentage) in enumerate(sorted_methods, 1):
        print(f"{i}. {method:<30}: {percentage:5.1f}%")
    
    print()

def main():
    """Main analysis function."""
    print("Integer Encoding Benchmark Analysis")
    print("===================================\n")
    
    # Path to the CSV file
    csv_path = "integer_encoding_benchmark_results/integer_encoding_benchmark_results.csv"
    
    # Try alternative path if not found
    if not os.path.exists(csv_path):
        csv_path = "src/benchmarking/integer_encoding_benchmark_results/integer_encoding_benchmark_results.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find CSV file at {csv_path}")
        return
    
    # Load results
    df = load_results(csv_path)
    if df is None:
        return
    
    print("Dataset Overview:")
    print(f"- Integer encoding methods: {list(df['timestamps_encoding_method'].unique())}")
    print(f"- Compression methods: {list(df['compression_method'].unique())}")
    print(f"- Error bounds tested: {list(df['error_bound'].unique())}\n")
    
    # Analyze best compression ratios
    best_compression = analyze_best_compression(df)
    compression_percentages, compression_total = calculate_percentages(best_compression)
    
    # Analyze best times
    best_time = analyze_best_time(df)
    time_percentages, time_total = calculate_percentages(best_time)
    
    # Analyze best compression sizes
    best_compression_size = analyze_best_compression_size(df)
    size_percentages, size_total = calculate_percentages(best_compression_size)
    
    # Analyze best combinations (functional + integer methods)
    best_combinations = analyze_best_combinations(df)
    combination_percentages, combination_total = calculate_percentages(best_combinations)
    
    # Analyze best time combinations (fastest overall time)
    best_time_combinations = analyze_best_time_combinations(df)
    time_combination_percentages, time_combination_total = calculate_percentages(best_time_combinations)
    
    # Analyze best balanced combinations (compression ratio + time)
    best_balanced_combinations = analyze_best_balanced_combinations(df)
    balanced_combination_percentages, balanced_combination_total = calculate_percentages(best_balanced_combinations)
    
    # Analyze dataset-specific performance
    dataset_winners = analyze_dataset_performance(df)
    
    # Analyze error bound impact
    error_bound_analysis = analyze_error_bound_impact(df)

    # Analyze average performance by integer encoding method
    avg_compression_by_method = analyze_average_total_compression(df)
    avg_time_by_method = analyze_average_time(df)
    avg_timestamps_compression_by_method = analyze_average_timestamps_compression(df)
    avg_timestamps_time_by_method = analyze_average_timestamps_time(df)

    # Print detailed statistics
    print_detailed_stats(compression_percentages, "Best Timestamp Compression Ratio by Method")
    print_detailed_stats(time_percentages, "Best Timestamp Encoding Time by Method")
    print_detailed_stats(size_percentages, "Best Timestamp Compression Size by Method")
    print_detailed_stats(combination_percentages, "Best Overall Combinations (Across All Scenarios)")
    print_detailed_stats(time_combination_percentages, "Best Overall Time Combinations (Fastest Total Time)")
    print_detailed_stats(balanced_combination_percentages, "Best Balanced Combinations (60% Ratio + 40% Time)")
    
    # Print average performance analysis
    print("\nAverage Performance by Integer Encoding Method")
    print("==============================================")
    print("\nAverage Total Compression Ratio:")
    print("-" * 35)
    sorted_compression = sorted(avg_compression_by_method.items(), key=lambda x: x[1], reverse=True)
    for i, (method, avg_ratio) in enumerate(sorted_compression, 1):
        print(f"{i}. {method:<30}: {avg_ratio:.4f}")
    
    print("\nAverage Total Processing Time:")
    print("-" * 34)
    sorted_time = sorted(avg_time_by_method.items(), key=lambda x: x[1])
    for i, (method, avg_time) in enumerate(sorted_time, 1):
        print(f"{i}. {method:<30}: {avg_time:.3f} seconds")
    
    print("\nAverage Timestamps Compression Ratio:")
    print("-" * 41)
    sorted_timestamps_compression = sorted(avg_timestamps_compression_by_method.items(), key=lambda x: x[1], reverse=True)
    for i, (method, avg_ratio) in enumerate(sorted_timestamps_compression, 1):
        print(f"{i}. {method:<30}: {avg_ratio:.4f}")
    
    print("\nAverage Timestamps Encoding Time:")
    print("-" * 37)
    sorted_timestamps_time = sorted(avg_timestamps_time_by_method.items(), key=lambda x: x[1])
    for i, (method, avg_time) in enumerate(sorted_timestamps_time, 1):
        print(f"{i}. {method:<30}: {avg_time:.4f} seconds")
    
    # Print dataset-specific analysis
    print("\nDataset-Specific Performance Analysis")
    print("====================================")
    for dataset, winners in dataset_winners.items():
        print(f"\n{dataset.upper()}:")
        print(f"  Best Compression: {winners['best_compression']}")
        print(f"  Best Time: {winners['best_time']}")
        print(f"  Most Consistent: {winners['most_consistent']} (wins {winners['consistency_score']}/3 error bounds)")
    
    # Print error bound impact analysis
    print("\nError Bound Impact Analysis")
    print("===========================")
    for error_bound in sorted(error_bound_analysis.keys()):
        analysis = error_bound_analysis[error_bound]
        print(f"\nError Bound {error_bound}:")
        print(f"  Avg Compression Ratio: {analysis['avg_compression_ratio']:.3f}")
        print(f"  Avg Processing Time: {analysis['avg_time_seconds']:.1f} seconds")
        print(f"  Avg Compressed Size: {analysis['avg_size_mb']:.1f} MB")
        print(f"  Most Popular Combo: {analysis['most_popular_combo']}")
    
    # Create visualizations
    print("Creating visualizations...")
    
    create_pie_chart(
        compression_percentages,
        "Best Timestamp Compression Ratio by Method",
        "best_timestamp_compression_ratio.png"
    )
    
    create_pie_chart(
        time_percentages,
        "Best Timestamp Encoding Time by Method", 
        "best_timestamp_encoding_time.png"
    )
    
    create_pie_chart(
        size_percentages,
        "Best Timestamp Compression Size by Method",
        "best_timestamp_compression_size.png"
    )
    
    create_combination_chart(
        combination_percentages,
        "Best Overall Combinations (Functional + Integer Methods)",
        "best_method_combinations.png"
    )
    
    create_combination_chart(
        time_combination_percentages,
        "Best Overall Time Combinations (Fastest Total Time)",
        "best_time_combinations.png"
    )
    
    create_combination_chart(
        balanced_combination_percentages,
        "Best Balanced Combinations (60% Compression Ratio + 40% Time)",
        "best_balanced_combinations.png"
    )
    
    # Create dataset comparison visualization
    create_dataset_comparison_chart(dataset_winners)
    
    # Create error bound impact visualization
    create_error_bound_analysis_chart(error_bound_analysis)
    
    # Create average performance visualizations
    create_average_compression_chart(
        avg_compression_by_method,
        "Average Total Compression Ratio by Integer Encoding Method",
        "average_compression_by_method.png"
    )
    
    create_average_time_chart(
        avg_time_by_method,
        "Average Total Processing Time by Integer Encoding Method",
        "average_time_by_method.png"
    )
    
    create_timestamps_compression_chart(
        avg_timestamps_compression_by_method,
        "Average Timestamps Compression Ratio by Integer Encoding Method",
        "average_timestamps_compression_by_method.png"
    )
    
    create_timestamps_time_chart(
        avg_timestamps_time_by_method,
        "Average Timestamps Encoding Time by Integer Encoding Method",
        "average_timestamps_time_by_method.png"
    )
    
    print("\nAnalysis completed!")
    print(f"Analyzed {compression_total} different test combinations")
    print("Generated visualizations in analyse_results_diagrams/ folder:")
    print("- analyse_results_diagrams/best_timestamp_compression_ratio.png")
    print("- analyse_results_diagrams/best_timestamp_encoding_time.png")
    print("- analyse_results_diagrams/best_timestamp_compression_size.png")
    print("- analyse_results_diagrams/best_method_combinations.png")
    print("- analyse_results_diagrams/best_time_combinations.png")
    print("- analyse_results_diagrams/best_balanced_combinations.png")
    print("- analyse_results_diagrams/dataset_performance_comparison.png")
    print("- analyse_results_diagrams/error_bound_impact_analysis.png")
    print("- analyse_results_diagrams/average_compression_by_method.png")
    print("- analyse_results_diagrams/average_time_by_method.png")
    print("- analyse_results_diagrams/average_timestamps_compression_by_method.png")
    print("- analyse_results_diagrams/average_timestamps_time_by_method.png")

if __name__ == "__main__":
    main()
