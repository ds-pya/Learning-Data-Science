import torch

labels = ["O", "B-A", "I-A", "B-B", "I-B", "B-C", "I-C", "B-D", "I-D"]
n_labels = len(labels)

# 초기 전환 비용 (0)
transition = torch.zeros(n_labels, n_labels)

def parse_tag(tag):
    """태그 분리: (bio, type)"""
    if tag == "O":
        return "O", None
    bio, ent_type = tag.split("-", 1)
    return bio, ent_type

# BIO 규칙 위반 → 100으로
for i, from_tag in enumerate(labels):
    from_bio, from_type = parse_tag(from_tag)
    for j, to_tag in enumerate(labels):
        to_bio, to_type = parse_tag(to_tag)

        invalid = False
        # 1. O → I-X 불가
        if from_bio == "O" and to_bio == "I":
            invalid = True
        # 2. B-X → I-Y (X != Y) 불가
        elif from_bio == "B" and to_bio == "I" and from_type != to_type:
            invalid = True
        # 3. I-X → I-Y (X != Y) 불가
        elif from_bio == "I" and to_bio == "I" and from_type != to_type:
            invalid = True

        if invalid:
            transition[i, j] = 100

print(transition)