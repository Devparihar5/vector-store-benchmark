#!/usr/bin/env python3
"""
Analyze and visualize benchmark results for ChromaDB vs PGVector
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})

def load_and_analyze():
    """Load benchmark results and create comprehensive visualizations"""
    results_dir = "benchmark_results"
    charts_dir = os.path.join(results_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    # Load all benchmark results
    benchmark_files = {
        "insertion": os.path.join(results_dir, "insertion_benchmark.csv"),
        "query": os.path.join(results_dir, "query_benchmark.csv"),
        "memory": os.path.join(results_dir, "memory_usage_benchmark.csv"),
        "cpu": os.path.join(results_dir, "cpu_usage_benchmark.csv")
    }
    
    dataframes = {}
    for name, file_path in benchmark_files.items():
        if os.path.exists(file_path):
            dataframes[name] = pd.read_csv(file_path)
            print(f"Loaded {name} benchmark data")
        else:
            print(f"Warning: {file_path} not found")
    
    if not dataframes:
        print("No benchmark results found. Run benchmark.py first.")
        return
    
    # Create summary report
    create_summary_report(dataframes)
    
    # Create visualizations
    create_visualizations(dataframes, charts_dir)
    
    # Create detailed comparison charts
    create_detailed_comparison(dataframes, charts_dir)

def create_summary_report(dataframes):
    """Create a summary report of all benchmark results"""
    summary = []
    
    # Insertion performance summary
    if "insertion" in dataframes:
        df = dataframes["insertion"]
        for size in df["dataset_size"].unique():
            size_df = df[df["dataset_size"] == size]
            chroma_time = size_df[size_df["database"] == "ChromaDB"]["time_seconds"].values[0]
            pgvector_time = size_df[size_df["database"] == "PGVector"]["time_seconds"].values[0]
            speedup = pgvector_time / chroma_time if chroma_time > 0 else float('inf')
            
            summary.append({
                "Benchmark": "Insertion",
                "Dataset Size": size,
                "ChromaDB": chroma_time,
                "PGVector": pgvector_time,
                "Ratio (PGVector/ChromaDB)": 1/speedup,  # Inverse for consistency (lower is better)
                "Better Option": "PGVector" if pgvector_time < chroma_time else "ChromaDB"
            })
    
    # Query performance summary
    if "query" in dataframes:
        df = dataframes["query"]
        for size in df["dataset_size"].unique():
            size_df = df[df["dataset_size"] == size]
            chroma_time = size_df[size_df["database"] == "ChromaDB"]["time_seconds"].values[0]
            pgvector_time = size_df[size_df["database"] == "PGVector"]["time_seconds"].values[0]
            speedup = pgvector_time / chroma_time if chroma_time > 0 else float('inf')
            
            summary.append({
                "Benchmark": "Query",
                "Dataset Size": size,
                "ChromaDB": chroma_time,
                "PGVector": pgvector_time,
                "Ratio (PGVector/ChromaDB)": speedup,  # Higher is worse for PGVector
                "Better Option": "ChromaDB" if chroma_time < pgvector_time else "PGVector"
            })
        
    # Memory usage summary
    if "memory" in dataframes:
        df = dataframes["memory"]
        for size in df["dataset_size"].unique():
            size_df = df[df["dataset_size"] == size]
            chroma_memory = size_df[size_df["database"] == "ChromaDB"]["memory_mb"].values[0]
            pgvector_memory = size_df[size_df["database"] == "PGVector"]["memory_mb"].values[0]
            ratio = pgvector_memory / chroma_memory if chroma_memory > 0 else float('inf')
            
            summary.append({
                "Benchmark": "Memory Usage",
                "Dataset Size": size,
                "ChromaDB": chroma_memory,
                "PGVector": pgvector_memory,
                "Ratio (PGVector/ChromaDB)": ratio,
                "Better Option": "PGVector" if pgvector_memory < chroma_memory else "ChromaDB"
            })
    
    # CPU usage summary
    if "cpu" in dataframes:
        df = dataframes["cpu"]
        for size in df["dataset_size"].unique():
            size_df = df[df["dataset_size"] == size]
            chroma_cpu = size_df[size_df["database"] == "ChromaDB"]["cpu_percent"].values[0]
            pgvector_cpu = size_df[size_df["database"] == "PGVector"]["cpu_percent"].values[0]
            ratio = pgvector_cpu / chroma_cpu if chroma_cpu > 0 else float('inf')
            
            summary.append({
                "Benchmark": "CPU Usage",
                "Dataset Size": size,
                "ChromaDB": chroma_cpu,
                "PGVector": pgvector_cpu,
                "Ratio (PGVector/ChromaDB)": ratio,
                "Better Option": "PGVector" if pgvector_cpu < chroma_cpu else "ChromaDB"
            })
    
    # Create summary DataFrame and save to CSV
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("benchmark_results/summary_report.csv", index=False)
    
    # Print summary
    print("\nSummary Report:")
    print(summary_df.to_string())
    
    # Create a more concise summary with recommendations
    create_recommendations(summary_df)

def create_recommendations(summary_df):
    """Create a concise summary with recommendations"""
    recommendations = []
    
    # Count wins for each database by dataset size
    for size in summary_df["Dataset Size"].unique():
        size_df = summary_df[summary_df["Dataset Size"] == size]
        chroma_wins = (size_df["Better Option"] == "ChromaDB").sum()
        pgvector_wins = (size_df["Better Option"] == "PGVector").sum()
        
        # Determine overall recommendation
        if chroma_wins > pgvector_wins:
            recommendation = "ChromaDB"
            reason = f"ChromaDB wins in {chroma_wins}/{len(size_df)} benchmarks"
        elif pgvector_wins > chroma_wins:
            recommendation = "PGVector"
            reason = f"PGVector wins in {pgvector_wins}/{len(size_df)} benchmarks"
        else:
            recommendation = "Either"
            reason = "Both perform similarly overall"
        
        # Get specific strengths
        chroma_strengths = size_df[size_df["Better Option"] == "ChromaDB"]["Benchmark"].tolist()
        pgvector_strengths = size_df[size_df["Better Option"] == "PGVector"]["Benchmark"].tolist()
        
        recommendations.append({
            "Dataset Size": size,
            "Recommendation": recommendation,
            "Reason": reason,
            "ChromaDB Strengths": ", ".join(chroma_strengths) if chroma_strengths else "None",
            "PGVector Strengths": ", ".join(pgvector_strengths) if pgvector_strengths else "None"
        })
    
    # Create recommendations DataFrame and save to CSV
    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df.to_csv("benchmark_results/recommendations.csv", index=False)
    
    # Print recommendations
    print("\nRecommendations:")
    print(recommendations_df.to_string())

def create_visualizations(dataframes, charts_dir):
    """Create visualizations for benchmark results"""
    # Performance comparison chart
    plt.figure(figsize=(15, 10))
    
    # Insertion subplot
    if "insertion" in dataframes:
        plt.subplot(2, 2, 1)
        insertion_df = dataframes["insertion"]
        for db in insertion_df["database"].unique():
            data = insertion_df[insertion_df["database"] == db]
            plt.plot(data["dataset_size"], data["time_seconds"], marker="o", label=db)
        plt.xlabel("Dataset Size")
        plt.ylabel("Time (seconds)")
        plt.title("Insertion Performance")
        plt.legend()
        plt.grid(True)
        plt.xscale('log')  # Log scale for better visualization
    
    # Query subplot
    if "query" in dataframes:
        plt.subplot(2, 2, 2)
        query_df = dataframes["query"]
        for db in query_df["database"].unique():
            data = query_df[query_df["database"] == db]
            plt.plot(data["dataset_size"], data["time_seconds"], marker="o", label=db)
        plt.xlabel("Dataset Size")
        plt.ylabel("Time (seconds)")
        plt.title("Query Performance")
        plt.legend()
        plt.grid(True)
        plt.xscale('log')  # Log scale for better visualization
    
    # Memory usage subplot
    if "memory" in dataframes:
        plt.subplot(2, 2, 3)
        memory_df = dataframes["memory"]
        for db in memory_df["database"].unique():
            data = memory_df[memory_df["database"] == db]
            plt.plot(data["dataset_size"], data["memory_mb"], marker="o", label=db)
        plt.xlabel("Dataset Size")
        plt.ylabel("Memory Usage (MB)")
        plt.title("Memory Usage")
        plt.legend()
        plt.grid(True)
        plt.xscale('log')  # Log scale for better visualization
    
    # CPU usage subplot
    if "cpu" in dataframes:
        plt.subplot(2, 2, 4)
        cpu_df = dataframes["cpu"]
        for db in cpu_df["database"].unique():
            data = cpu_df[cpu_df["database"] == db]
            plt.plot(data["dataset_size"], data["cpu_percent"], marker="o", label=db)
        plt.xlabel("Dataset Size")
        plt.ylabel("CPU Usage (%)")
        plt.title("CPU Usage")
        plt.legend()
        plt.grid(True)
        plt.xscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "performance_comparison.png"))
    
    # Create radar chart for overall comparison
    create_radar_chart(dataframes, charts_dir)

def create_detailed_comparison(dataframes, charts_dir):
    """Create detailed comparison charts for each metric"""
    # Bar charts for each dataset size
    for metric, df_name in [
        ("Insertion Time (s)", "insertion"), 
        ("Query Time (s)", "query"), 
        ("Memory Usage (MB)", "memory"), 
        ("CPU Usage (%)", "cpu")
    ]:
        if df_name in dataframes:
            df = dataframes[df_name]
            plt.figure(figsize=(12, 8))
            
            # Reshape data for grouped bar chart
            pivot_df = df.pivot(index="dataset_size", columns="database", values=df.columns[-1])
            
            # Create bar chart
            ax = pivot_df.plot(kind="bar", width=0.7)
            
            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', padding=3)
            
            plt.xlabel("Dataset Size")
            plt.ylabel(metric)
            plt.title(f"{metric} Comparison")
            plt.grid(axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, f"{df_name}_comparison_bars.png"))
    
    # Create scaling charts to show how each database scales with dataset size
    create_scaling_charts(dataframes, charts_dir)

def create_scaling_charts(dataframes, charts_dir):
    """Create charts showing how metrics scale with dataset size"""
    plt.figure(figsize=(15, 10))
    
    # Insertion time scaling
    if "insertion" in dataframes:
        plt.subplot(2, 2, 1)
        df = dataframes["insertion"]
        for db in df["database"].unique():
            data = df[df["database"] == db]
            # Fit a line to see scaling behavior
            x = data["dataset_size"]
            y = data["time_seconds"]
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x, y, 'o', label=f"{db} (actual)")
            plt.plot(x, p(x), '--', label=f"{db} (trend)")
        plt.xlabel("Dataset Size")
        plt.ylabel("Time (seconds)")
        plt.title("Insertion Time Scaling")
        plt.legend()
        plt.grid(True)
    
    # Query time scaling
    if "query" in dataframes:
        plt.subplot(2, 2, 2)
        df = dataframes["query"]
        for db in df["database"].unique():
            data = df[df["database"] == db]
            # Fit a line to see scaling behavior
            x = data["dataset_size"]
            y = data["time_seconds"]
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x, y, 'o', label=f"{db} (actual)")
            plt.plot(x, p(x), '--', label=f"{db} (trend)")
        plt.xlabel("Dataset Size")
        plt.ylabel("Time (seconds)")
        plt.title("Query Time Scaling")
        plt.legend()
        plt.grid(True)
    
    # Memory usage scaling
    if "memory" in dataframes:
        plt.subplot(2, 2, 3)
        df = dataframes["memory"]
        for db in df["database"].unique():
            data = df[df["database"] == db]
            # Fit a line to see scaling behavior
            x = data["dataset_size"]
            y = data["memory_mb"]
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x, y, 'o', label=f"{db} (actual)")
            plt.plot(x, p(x), '--', label=f"{db} (trend)")
        plt.xlabel("Dataset Size")
        plt.ylabel("Memory (MB)")
        plt.title("Memory Usage Scaling")
        plt.legend()
        plt.grid(True)
    
    # CPU usage scaling
    if "cpu" in dataframes:
        plt.subplot(2, 2, 4)
        df = dataframes["cpu"]
        for db in df["database"].unique():
            data = df[df["database"] == db]
            # Fit a line to see scaling behavior
            x = data["dataset_size"]
            y = data["cpu_percent"]
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x, y, 'o', label=f"{db} (actual)")
            plt.plot(x, p(x), '--', label=f"{db} (trend)")
        plt.xlabel("Dataset Size")
        plt.ylabel("CPU Usage (%)")
        plt.title("CPU Usage Scaling")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "scaling_analysis.png"))

def create_radar_chart(dataframes, charts_dir):
    """Create a radar chart for overall comparison"""
    import numpy as np
    
    # Prepare data for radar chart
    categories = []
    chroma_values = []
    pgvector_values = []
    
    # For each metric, we'll normalize values between 0 and 1
    # where 1 is always better performance (faster/less resource usage)
    
    # Insertion time (normalized)
    if "insertion" in dataframes:
        df = dataframes["insertion"]
        max_size = df["dataset_size"].max()
        size_df = df[df["dataset_size"] == max_size]
        
        chroma_time = size_df[size_df["database"] == "ChromaDB"]["time_seconds"].values[0]
        pgvector_time = size_df[size_df["database"] == "PGVector"]["time_seconds"].values[0]
        max_time = max(chroma_time, pgvector_time)
        
        categories.append("Insertion Speed")
        # Lower time is better, so normalize and invert
        chroma_values.append(1 - (chroma_time / max_time))
        pgvector_values.append(1 - (pgvector_time / max_time))
    
    # Query time (normalized)
    if "query" in dataframes:
        df = dataframes["query"]
        max_size = df["dataset_size"].max()
        size_df = df[df["dataset_size"] == max_size]
        
        chroma_time = size_df[size_df["database"] == "ChromaDB"]["time_seconds"].values[0]
        pgvector_time = size_df[size_df["database"] == "PGVector"]["time_seconds"].values[0]
        max_time = max(chroma_time, pgvector_time)
        
        categories.append("Query Speed")
        # Lower time is better, so normalize and invert
        chroma_values.append(1 - (chroma_time / max_time))
        pgvector_values.append(1 - (pgvector_time / max_time))
    
    # Memory efficiency (normalized)
    if "memory" in dataframes:
        df = dataframes["memory"]
        max_size = df["dataset_size"].max()
        size_df = df[df["dataset_size"] == max_size]
        
        chroma_memory = size_df[size_df["database"] == "ChromaDB"]["memory_mb"].values[0]
        pgvector_memory = size_df[size_df["database"] == "PGVector"]["memory_mb"].values[0]
        max_memory = max(chroma_memory, pgvector_memory)
        
        categories.append("Memory Efficiency")
        # Lower memory is better, so normalize and invert
        chroma_values.append(1 - (chroma_memory / max_memory))
        pgvector_values.append(1 - (pgvector_memory / max_memory))
    
    # CPU efficiency (normalized)
    if "cpu" in dataframes:
        df = dataframes["cpu"]
        max_size = df["dataset_size"].max()
        size_df = df[df["dataset_size"] == max_size]
        
        chroma_cpu = size_df[size_df["database"] == "ChromaDB"]["cpu_percent"].values[0]
        pgvector_cpu = size_df[size_df["database"] == "PGVector"]["cpu_percent"].values[0]
        max_cpu = max(chroma_cpu, pgvector_cpu)
        
        categories.append("CPU Efficiency")
        # Lower CPU usage is better, so normalize and invert
        chroma_values.append(1 - (chroma_cpu / max_cpu))
        pgvector_values.append(1 - (pgvector_cpu / max_cpu))
    
    # Create radar chart
    if categories:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Number of variables
        N = len(categories)
        
        # Angle of each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add values for each database
        chroma_values += chroma_values[:1]  # Close the loop
        pgvector_values += pgvector_values[:1]  # Close the loop
        
        # Plot with thicker lines and larger markers
        ax.plot(angles, chroma_values, 'o-', linewidth=3, markersize=10, label='ChromaDB')
        ax.fill(angles, chroma_values, alpha=0.25)
        ax.plot(angles, pgvector_values, 'o-', linewidth=3, markersize=10, label='PGVector')
        ax.fill(angles, pgvector_values, alpha=0.25)
        
        # Set category labels with better positioning
        plt.xticks(angles[:-1], categories, size=14)
        
        # Add value labels at each point
        for i, (angle, value) in enumerate(zip(angles[:-1], chroma_values[:-1])):
            ax.text(angle, value + 0.05, f"{value:.2f}", 
                    horizontalalignment='center', size=12, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        for i, (angle, value) in enumerate(zip(angles[:-1], pgvector_values[:-1])):
            ax.text(angle, value - 0.1, f"{value:.2f}", 
                    horizontalalignment='center', size=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        
        # Set y-axis limits to ensure consistent scale
        ax.set_ylim(0, 1.2)
        
        # Add gridlines
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], angle=45)
        
        # Add legend with better positioning and larger font
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=14)
        
        # Add a note explaining the chart
        plt.figtext(0.5, 0.01, 
                   "Note: Higher values indicate better performance.\n"
                   "All metrics are normalized where 1.0 represents the best performance.", 
                   ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        plt.title('ChromaDB vs PGVector Performance Comparison', size=18, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "radar_comparison.png"), bbox_inches='tight', dpi=300)
        
        # Also create a version with absolute values for better interpretation
        create_absolute_radar_chart(dataframes, charts_dir)

def create_summary_table_visualization(dataframes, charts_dir):
    """Create a visual summary table"""
    # Collect data for the table
    table_data = []
    metrics = ["Insertion Time (s)", "Query Time (s)", "Memory Usage (MB)", "CPU Usage (%)"]
    units = ["s", "s", "MB", "%"]
    
    # Get the largest dataset size
    max_size = 0
    for df_name in dataframes:
        df = dataframes[df_name]
        max_size = max(max_size, df["dataset_size"].max())
    
    # Collect data for each metric
    for i, (metric, df_name, unit) in enumerate([
        ("Insertion Time", "insertion", "s"), 
        ("Query Time", "query", "s"), 
        ("Memory Usage", "memory", "MB"), 
        ("CPU Usage", "cpu", "%")
    ]):
        if df_name in dataframes:
            df = dataframes[df_name]
            size_df = df[df["dataset_size"] == max_size]
            
            chroma_val = size_df[size_df["database"] == "ChromaDB"].iloc[0, -1]
            pgvector_val = size_df[size_df["database"] == "PGVector"].iloc[0, -1]
            
            # Calculate ratio and determine winner
            ratio = pgvector_val / chroma_val if chroma_val > 0 else float('inf')
            
            # For all metrics, lower is better
            winner = "ChromaDB" if chroma_val < pgvector_val else "PGVector"
            
            # Calculate percentage difference
            if winner == "ChromaDB":
                pct_diff = ((pgvector_val - chroma_val) / pgvector_val) * 100
            else:
                pct_diff = ((chroma_val - pgvector_val) / chroma_val) * 100
            
            table_data.append([
                f"{metric} ({unit})", 
                chroma_val, 
                pgvector_val, 
                f"{ratio:.2f}x", 
                winner,
                f"{pct_diff:.1f}% better"
            ])
    
    # Create a visual table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table with improved formatting
    table = ax.table(
        cellText=[[row[0], f"{row[1]:.2f}", f"{row[2]:.2f}", row[3], row[4], row[5]] for row in table_data],
        colLabels=["Metric", "ChromaDB", "PGVector", "Ratio (PG/Chroma)", "Winner", "Difference"],
        cellLoc='center',
        loc='center',
        cellColours=[
            [
                '#f0f0f0',  # Metric column
                '#d6eaf8' if row[4] == "ChromaDB" else '#f5eef8',  # ChromaDB value
                '#f5eef8' if row[4] == "PGVector" else '#d6eaf8',  # PGVector value
                '#f0f0f0',  # Ratio
                '#d5f5e3' if row[4] == "ChromaDB" else '#fadbd8',  # Winner
                '#d5f5e3' if row[4] == "ChromaDB" else '#fadbd8'   # Difference
            ]
            for row in table_data
        ]
    )
    
    # Format the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    
    # Adjust column widths
    for (row, col), cell in table._cells.items():
        if col == 0:  # Metric column
            cell.set_width(0.25)
        elif col in [1, 2]:  # Value columns
            cell.set_width(0.15)
        elif col == 3:  # Ratio column
            cell.set_width(0.15)
        else:  # Winner and difference columns
            cell.set_width(0.15)
    
    plt.title(f'Performance Summary (Dataset Size: {max_size})', fontsize=18)
    
    # Add a note explaining the table
    plt.figtext(0.5, 0.01, 
               "Note: For all metrics, lower values are better.\n"
               "Ratio shows PGVector/ChromaDB, so values > 1 mean ChromaDB is better.", 
               ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "summary_table.png"), bbox_inches='tight', dpi=300)

def create_absolute_radar_chart(dataframes, charts_dir):
    """Create a radar chart showing the ratio of performance between databases"""
    import numpy as np
    
    # Prepare data for radar chart
    categories = []
    ratio_values = []  # PGVector/ChromaDB ratio
    better_db = []  # Which DB is better for each metric
    
    # Insertion time ratio
    if "insertion" in dataframes:
        df = dataframes["insertion"]
        max_size = df["dataset_size"].max()
        size_df = df[df["dataset_size"] == max_size]
        
        chroma_time = size_df[size_df["database"] == "ChromaDB"]["time_seconds"].values[0]
        pgvector_time = size_df[size_df["database"] == "PGVector"]["time_seconds"].values[0]
        
        # Calculate ratio (PGVector/ChromaDB)
        ratio = pgvector_time / chroma_time if chroma_time > 0 else float('inf')
        
        categories.append("Insertion Time")
        ratio_values.append(ratio)
        better_db.append("ChromaDB" if ratio > 1 else "PGVector")
    
    # Query time ratio
    if "query" in dataframes:
        df = dataframes["query"]
        max_size = df["dataset_size"].max()
        size_df = df[df["dataset_size"] == max_size]
        
        chroma_time = size_df[size_df["database"] == "ChromaDB"]["time_seconds"].values[0]
        pgvector_time = size_df[size_df["database"] == "PGVector"]["time_seconds"].values[0]
        
        # Calculate ratio (PGVector/ChromaDB)
        ratio = pgvector_time / chroma_time if chroma_time > 0 else float('inf')
        
        categories.append("Query Time")
        ratio_values.append(ratio)
        better_db.append("ChromaDB" if ratio > 1 else "PGVector")
    
    # Memory usage ratio
    if "memory" in dataframes:
        df = dataframes["memory"]
        max_size = df["dataset_size"].max()
        size_df = df[df["dataset_size"] == max_size]
        
        chroma_memory = size_df[size_df["database"] == "ChromaDB"]["memory_mb"].values[0]
        pgvector_memory = size_df[size_df["database"] == "PGVector"]["memory_mb"].values[0]
        
        # Calculate ratio (PGVector/ChromaDB)
        ratio = pgvector_memory / chroma_memory if chroma_memory > 0 else float('inf')
        
        categories.append("Memory Usage")
        ratio_values.append(ratio)
        better_db.append("ChromaDB" if ratio > 1 else "PGVector")
    
    # CPU usage ratio
    if "cpu" in dataframes:
        df = dataframes["cpu"]
        max_size = df["dataset_size"].max()
        size_df = df[df["dataset_size"] == max_size]
        
        chroma_cpu = size_df[size_df["database"] == "ChromaDB"]["cpu_percent"].values[0]
        pgvector_cpu = size_df[size_df["database"] == "PGVector"]["cpu_percent"].values[0]
        
        # Calculate ratio (PGVector/ChromaDB)
        ratio = pgvector_cpu / chroma_cpu if chroma_cpu > 0 else float('inf')
        
        categories.append("CPU Usage")
        ratio_values.append(ratio)
        better_db.append("ChromaDB" if ratio > 1 else "PGVector")
    
    # Create bar chart for ratios
    if categories:
        plt.figure(figsize=(12, 8))
        
        # Create bars
        bars = plt.bar(categories, ratio_values, color=['green' if db == "ChromaDB" else 'red' for db in better_db])
        
        # Add a horizontal line at y=1 (equal performance)
        plt.axhline(y=1, color='black', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}x', ha='center', va='bottom', fontsize=12)
        
        # Add which DB is better
        for i, (bar, db) in enumerate(zip(bars, better_db)):
            plt.text(bar.get_x() + bar.get_width()/2., 0.1,
                    f'{db} better', ha='center', va='bottom', 
                    fontsize=10, rotation=90, color='white')
        
        plt.ylabel('Ratio (PGVector / ChromaDB)')
        plt.title('Performance Ratio: PGVector vs ChromaDB\n(Higher ratio means ChromaDB is better)', fontsize=16)
        
        # Add explanation
        plt.figtext(0.5, 0.01, 
                   "Note: Bars above 1.0 mean ChromaDB performs better.\n"
                   "Bars below 1.0 mean PGVector performs better.", 
                   ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "performance_ratio.png"), bbox_inches='tight', dpi=300)
        
        # Create a summary table visualization
        create_summary_table_visualization(dataframes, charts_dir)

if __name__ == "__main__":
    print("Analyzing benchmark results...")
    load_and_analyze()
    print("Analysis complete. Results saved in 'benchmark_results/charts' directory.")
