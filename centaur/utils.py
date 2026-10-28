import torch
import numpy as np


def get_probs(model, tokenizer, input_text, valid_tokens, temp=1.0):
    if temp < 0:
        raise ValueError("Temperature must be positive")

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]

    valid_t_ids = []
    for t in valid_tokens:
        id = tokenizer.encode(t, add_special_tokens=False)
        id_s = tokenizer.encode(" " + t, add_special_tokens=False)

        if len(id) == 1:
            valid_t_ids.append(id[0])
        elif len(id_s) == 1:
            valid_t_ids.append(id_s[0])

    ids_tensor = torch.tensor(valid_t_ids).to(model.device)
    t_logits = logits.index_select(dim=-1, index=ids_tensor).to(torch.float64)

    if temp == 0:
        probs = torch.zeros_like(t_logits)
        probs[0][torch.argmax(t_logits)] = 1.0
    else:
        probs = torch.nn.functional.softmax(t_logits / temp, dim=-1)

    return probs


def compute_regret(hist, m_hist):
    regret = 0
    for i, choice in enumerate(hist):
        regret += max(m_hist[i]) - m_hist[i][choice - 1]

    return regret


def gen_features(bandit):
    feats = []
    str_feats = ""
    for i in range(bandit.narms):
        str_feats += f"Button {i + 1} features: "
        ft = np.random.randint(0, 2, bandit.nfeats)
        feats.append(ft)
        str_feats += str(ft) + "\n"

    return feats, str_feats


def gen_independent_features(bandit):
    feats = []
    str_feats = ""
    one_index = np.random.randint(0, bandit.narms)
    for i in range(bandit.narms):
        str_feats += f"Button {i + 1} features: "
        if i == one_index:
            ft = np.random.randint(0, 2, bandit.nfeats)
            ft[2] = 1
        else:
            ft = np.random.randint(0, 2, bandit.nfeats)
            ft[2] = 0
        feats.append(ft)
        str_feats += str(ft) + "\n"

    return feats, str_feats


if __name__ == "__main__":
    # test out regret computation
    hist = [0, 0, 0, 0, 0]
    m_hist = [
        [10, 20, 30, 40],
        [10, 20, 30, 40],
        [10, 20, 30, 40],
        [10, 20, 30, 40],
        [10, 20, 30, 40],
    ]
    regret = compute_regret(hist, m_hist)
    print("Cumulative Regret:", regret)
