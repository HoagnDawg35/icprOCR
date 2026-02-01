from collections import defaultdict
import numpy as np

def print_character_stats(cm, chars):
    """Prints a detailed table of character-wise statistics."""
    plot_chars = sorted(list(chars))
    
    print("\n📊 Character-wise Error Statistics:")
    header = f"{'Char':<6} | {'Total':<6} | {'Correct':<8} | {'Sub':<6} | {'Del':<6} | {'Ins':<6} | {'Err%':<6}"
    print(header)
    print("-" * len(header))
    
    total_gt_all = 0
    total_correct_all = 0
    total_sub_all = 0
    total_del_all = 0
    total_ins_all = 0
    
    for c in plot_chars:
        # Total GT occurrences = Matches + Substitutions + Deletions
        correct = cm[c][c]
        deleted = cm[c]['<DEL>']
        substituted = sum(count for pred, count in cm[c].items() if pred != c and pred != '<DEL>' and pred != '<INS>')
        total_gt = correct + deleted + substituted
        
        inserted = cm['<INS>'][c]
        
        error_rate = ((total_gt - correct + inserted) / total_gt * 100) if total_gt > 0 else 0
        
        if total_gt > 0 or inserted > 0:
            print(f"{c:<6} | {total_gt:<6} | {correct:<8} | {substituted:<6} | {deleted:<6} | {inserted:<6} | {error_rate:>5.1f}%")
            
            total_gt_all += total_gt
            total_correct_all += correct
            total_sub_all += substituted
            total_del_all += deleted
            total_ins_all += inserted

    print("-" * len(header))
    total_err_rate = ((total_gt_all - total_correct_all + total_ins_all) / total_gt_all * 100) if total_gt_all > 0 else 0
    print(f"{'TOTAL':<6} | {total_gt_all:<6} | {total_correct_all:<8} | {total_sub_all:<6} | {total_del_all:<6} | {total_ins_all:<6} | {total_err_rate:>5.1f}%")
    print("")

# Dummy data
cm = defaultdict(lambda: defaultdict(int))
chars = ['A', 'B', '0']
# A: 10 total. 8 correct, 1 sub to B, 1 deleted. 2 inserts of A.
cm['A']['A'] = 8
cm['A']['B'] = 1
cm['A']['<DEL>'] = 1
cm['<INS>']['A'] = 2

# B: 5 total. 5 correct.
cm['B']['B'] = 5

# 0: 20 total. 15 correct, 5 sub to O (not in chars but should work)
cm['0']['0'] = 15
cm['0']['O'] = 5

print_character_stats(cm, chars)
