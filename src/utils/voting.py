from collections import defaultdict, Counter

def char_level_voting(preds):
    """
    preds: List[(text, conf)]
    """
    texts = [t for t, _ in preds]
    confs = [c for _, c in preds]

    max_len = max(len(t) for t in texts)

    # pad texts
    padded = [list(t.ljust(max_len, "<")) for t in texts]

    final_text = []
    final_conf = []

    for i in range(max_len):
        char_votes = defaultdict(list)

        for text, conf in preds:
            ch = text[i] if i < len(text) else "<"
            char_votes[ch].append(conf)

        # vote by count → confidence
        best_char, best_confs = sorted(
            char_votes.items(),
            key=lambda x: (len(x[1]), sum(x[1]) / len(x[1])),
            reverse=True
        )[0]

        if best_char != "<":
            final_text.append(best_char)
            final_conf.append(sum(best_confs) / len(best_confs))

    avg_conf = sum(final_conf) / len(final_conf) if final_conf else 0.0
    return "".join(final_text), avg_conf