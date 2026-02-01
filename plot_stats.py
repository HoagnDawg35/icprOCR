import os
import argparse
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_stats(csv_path, output_dir):
    """Plots character-wise statistics from a CSV file."""
    chars = []
    totals = []
    corrects = []
    error_counts = [] # Total errors (Sub + Del + Ins)

    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Char'] == 'TOTAL':
                continue
            chars.append(row['Char'])
            totals.append(int(row['Total']))
            corrects.append(int(row['Correct']))
            # Total errors = Sub + Del + Ins
            err_val = int(row['Sub']) + int(row['Del']) + int(row['Ins'])
            error_counts.append(err_val)

    if not chars:
        print("⚠️ No data found in CSV (besides TOTAL).")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Character Frequency (Total vs Correct)
    plt.figure(figsize=(15, 8))
    x = np.arange(len(chars))
    width = 0.35

    plt.bar(x - width/2, totals, width, label='Total GT', color='skyblue')
    plt.bar(x + width/2, corrects, width, label='Correctly Recognized', color='salmon')

    plt.xlabel('Character')
    plt.ylabel('Count')
    plt.title('Character Frequency and Recognition Accuracy')
    plt.xticks(x, chars)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    freq_path = os.path.join(output_dir, "char_frequency.png")
    plt.savefig(freq_path)
    plt.close()
    print(f"✅ Frequency plot saved to {freq_path}")

    # Plot 2: Character-wise Error Count (Absolute)
    plt.figure(figsize=(15, 8))
    plt.bar(chars, error_counts, color='orange', alpha=0.8)
    
    plt.xlabel('Character')
    plt.ylabel('Total Error Count (Sub + Del + Ins)')
    plt.title('Character-wise Error Counts (Absolute Numbers)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add labels on top of bars
    for i, count in enumerate(error_counts):
        plt.text(i, count + 0.1, f"{count}", ha='center', fontsize=9)

    err_path = os.path.join(output_dir, "char_error_counts.png")
    plt.savefig(err_path)
    plt.close()
    print(f"✅ Error count plot saved to {err_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot Character OCR statistics from CSV")
    parser.add_argument("--csv", type=str, required=True, help="Path to char_stats.csv")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (defaults to same as CSV)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.csv)

    plot_stats(args.csv, args.output_dir)
    print("🏁 Finished plotting.")

if __name__ == "__main__":
    main()
