import Levenshtein

def get_alignment_ops(gt, pred):
    return Levenshtein.editops(gt, pred)

def test_alignment(gt, pred):
    ops = get_alignment_ops(gt, pred)
    print(f"GT: {gt} | Pred: {pred}")
    print(f"Ops: {ops}")
    
    gt_used = [False] * len(gt)
    pred_used = [False] * len(pred)
    
    for op, i, j in ops:
        if op == 'replace':
            print(f"  REPLACE: {gt[i]} -> {pred[j]}")
            gt_used[i] = True
            pred_used[j] = True
        elif op == 'delete':
            print(f"  DELETE: {gt[i]}")
            gt_used[i] = True
        elif op == 'insert':
            print(f"  INSERT: {pred[j]}")
            pred_used[j] = True
            
    for i in range(len(gt)):
        if not gt_used[i]:
            print(f"  MATCH: {gt[i]}")

test_alignment("AYG7579", "AY07579") # 0 instead of G
test_alignment("ODE3320", "ODE332")  # Missing 0
test_alignment("DRC6817", "DRC68177") # Extra 7
