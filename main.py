import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from openai import OpenAI
import time
import re
from zhipuai import ZhipuAI
import os
from transformers import AutoTokenizer, AutoModel
import pickle
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity

import re
import json

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
random.seed(42)

from scipy.stats import entropy
from sklearn.metrics import accuracy_score, mutual_info_score, precision_score, recall_score, f1_score, roc_auc_score

import json

STATE_PATH = "all_agent_training_state.json"

from sklearn.cluster import AgglomerativeClustering
from scipy.stats import mode

from sklearn.cluster import AgglomerativeClustering
import numpy as np
from collections import defaultdict
from copy import deepcopy

import copy

def hard_downweight_nudging(top_leader_agents, all_leader_results, gt_batch, lr=0.05, wmin=0.0, wmax=1.0):

    y = np.array(gt_batch, dtype=int)
    for i, leader in enumerate(top_leader_agents):
        preds_i = np.array([r.get("Decision", 1) for r in all_leader_results[i]], dtype=int)
        if len(preds_i) == 0:
            continue
        acc_i = float(np.mean(preds_i == y))

        for k in range(len(getattr(leader, "subordinate_agents", []))):
            old_w = float(leader.confidence.get(k, 1.0))
            new_w = (1.0 - lr) * old_w + lr * acc_i
            leader.confidence[k] = float(np.clip(new_w, wmin, wmax))

def metrics_from_records(batch_records):
    all_y_true, all_y_pred = [], []
    for idx in sorted(batch_records.keys(), key=lambda x: int(x)):
        all_y_true.extend(batch_records[idx]["y_true"])
        all_y_pred.extend(batch_records[idx]["y_pred"])
    acc = accuracy_score(all_y_true, all_y_pred)
    p   = precision_score(all_y_true, all_y_pred, zero_division=0)
    r   = recall_score(all_y_true, all_y_pred, zero_division=0)
    f   = f1_score(all_y_true, all_y_pred, zero_division=0)
    try:
        auc = roc_auc_score(all_y_true, all_y_pred)
    except Exception:
        auc = 0.0
    return acc, p, r, f, auc

def leader_k_metrics_from_records(batch_records, k):
    all_y_true, all_y_pred = [], []
    for idx in sorted(batch_records.keys(), key=lambda x: int(x)):
        y_true = batch_records[idx]["y_true"]
        leader_results = batch_records[idx]["leader_results"]
        preds_k = [r.get("Decision", 1) for r in leader_results[k]]
        all_y_true.extend(y_true)
        all_y_pred.extend(preds_k)
    acc = accuracy_score(all_y_true, all_y_pred)
    p   = precision_score(all_y_true, all_y_pred, zero_division=0)
    r   = recall_score(all_y_true, all_y_pred, zero_division=0)
    f   = f1_score(all_y_true, all_y_pred, zero_division=0)
    try:
        auc = roc_auc_score(all_y_true, all_y_pred)
    except Exception:
        auc = 0.0
    return acc, p, r, f, auc

def base_k_metrics_from_records(batch_records, k):
    """可选：计算第 k 个 BaseAgent 在测试集上的指标。"""
    all_y_true, all_y_pred = [], []
    for idx in sorted(batch_records.keys(), key=lambda x: int(x)):
        y_true = batch_records[idx]["y_true"]
        base_results = batch_records[idx]["base_results"]      # 形如: [num_base][batch]
        preds_k = [r.get("Decision", 1) for r in base_results[k]]
        all_y_true.extend(y_true)
        all_y_pred.extend(preds_k)
    acc = accuracy_score(all_y_true, all_y_pred)
    p   = precision_score(all_y_true, all_y_pred, zero_division=0)
    r   = recall_score(all_y_true, all_y_pred, zero_division=0)
    f   = f1_score(all_y_true, all_y_pred, zero_division=0)
    try:
        auc = roc_auc_score(all_y_true, all_y_pred)
    except Exception:
        auc = 0.0
    return acc, p, r, f, auc

def collect_all_leader_agents(agent_list):
    all_leaders = []
    for agent in agent_list:
        if isinstance(agent, LeaderAgent):
            all_leaders.append(agent)
            if hasattr(agent, 'subordinate_agents') and agent.subordinate_agents:
                sub_leaders = collect_all_leader_agents(agent.subordinate_agents)
                all_leaders.extend(sub_leaders)
    return all_leaders

def get_subordinate_results_for_leader(agent, user_batch, all_base_results_T, leader_results_dict):

    if not hasattr(agent, 'subordinate_agents') or not agent.subordinate_agents:
        return all_base_results_T
    results = []
    for sub in agent.subordinate_agents:
        if hasattr(sub, "agent_id") and sub.agent_id in leader_results_dict:
            results.append(leader_results_dict[sub.agent_id])
        else:
            # fallback
            results.append([{"Decision": 1, "Reason": "No cache."}] * len(user_batch))

    results_T = list(map(list, zip(*results)))
    return results_T


def save_decision_experience(experience, path="experience_decisionagent.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"experience": experience}, f, ensure_ascii=False, indent=2)

def load_decision_experience(path="experience_decisionagent.json"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("experience", [])
    else:
        return []


def print_team_structure(agent, indent=0):
    prefix = " " * indent
    print(f"{prefix}{agent.agent_id} ({type(agent).__name__})")
    if hasattr(agent, "subordinate_agents") and agent.subordinate_agents:
        for sub in agent.subordinate_agents:
            print_team_structure(sub, indent + 4)

def sample_reason_recursive(agent, user_id, decision):

    if not hasattr(agent, 'subordinate_agents') or not agent.subordinate_agents:
        # BaseAgent
        mem = getattr(agent, 'user_decision_map', {})  # ImprovedBaseAgent
        if user_id in mem and mem[user_id].get("Decision") == decision:
            return mem[user_id].get("Reason", "No reason")
        return None
    else:
        candidates = []
        for idx, sub in enumerate(agent.subordinate_agents):
            if hasattr(sub, "memory") and user_id in sub.memory:
                sub_decision = sub.memory[user_id].get("decision")
            elif hasattr(sub, "user_decision_map") and user_id in sub.user_decision_map:
                sub_decision = sub.user_decision_map[user_id].get("Decision")
            else:
                sub_decision = None
            if sub_decision == decision:
                reason = sample_reason_recursive(sub, user_id, decision)
                conf = getattr(sub, "confidence", 1.0)
                if isinstance(conf, dict):
                    conf = conf.get(idx, 1.0)
                candidates.append((reason, conf))
        candidates = [(r, c) for r, c in candidates if r is not None]
        if candidates:
            max_conf = max(candidates, key=lambda x: x[1])[1]
            best = [r for r, c in candidates if c == max_conf]
            return random.choice(best)
        return None

def export_team_structure(agent, layer=1):
    struct = {
        "name": getattr(agent, "name", getattr(agent, "agent_name", str(agent))),
        "type": type(agent).__name__,
        "agent_id": getattr(agent, "agent_id", None),
        "layer": layer
    }
    if hasattr(agent, "subordinate_agents") and agent.subordinate_agents:
        struct["members"] = [export_team_structure(sub, layer+1) for sub in agent.subordinate_agents]
    return struct

def structure_adaptive_search(leader_agents, all_leaders, base_agents, train_users, train_labels, val_users, val_labels):

    print("==== [Structure Search] Start ====")
    change_log = []
    f1_before = validate_leader_structure(
        leader_agents, val_users, val_labels, base_agents=base_agents
    )

    for leader in reversed(all_leaders): 
        # ---- prune ----
        old_leader = copy.deepcopy(leader)
        prune_cnt = prune_similar_agents(leader, val_users, val_labels)
        if prune_cnt > 0:
            new_f1 = validate_leader_structure(
                leader_agents, val_users, val_labels, base_agents=base_agents
            )
            if new_f1 < f1_before:
                leader.__dict__ = old_leader.__dict__
                print(f"[{leader.agent_name}] prune {prune_cnt} agents, but F1 dropped ({f1_before:.4f} → {new_f1:.4f}), rollback.")
            else:
                f1_before = new_f1
                change_log.append(f"[{leader.agent_name}] prune {prune_cnt} agents (F1 ↑ {f1_before:.4f})")

        # ---- widen ----
        old_leader = copy.deepcopy(leader)
        widen_cnt = widen_leader_agent(leader, all_leaders, val_users, val_labels)
        if widen_cnt > 0:
            new_f1 = validate_leader_structure(
                leader_agents, val_users, val_labels, base_agents=base_agents
            )
            if new_f1 < f1_before:
                leader.__dict__ = old_leader.__dict__
                print(f"[{leader.agent_name}] widen add {widen_cnt} agents, but F1 dropped ({f1_before:.4f} → {new_f1:.4f}), rollback.")
            else:
                f1_before = new_f1
                change_log.append(f"[{leader.agent_name}] widen add {widen_cnt} agents (F1 ↑ {f1_before:.4f})")

    print("==== [Structure Search] Done ====")
    for log in change_log:
        print(log)

def _collect_decision_matrix_and_weights(leader, val_users):

    subs = getattr(leader, 'subordinate_agents', [])
    n = len(subs)
    decisions = []
    for ag in subs:
        udm = getattr(ag, 'user_decision_map', {})
        row = [udm[u]["Decision"] if u in udm else 1 for u in val_users]
        decisions.append(row)
    decisions = np.array(decisions, dtype=int) if decisions else np.zeros((0, len(val_users)), dtype=int)
    # 权重按下标映射补齐
    weights = np.array([leader.confidence.get(i, 1.0) for i in range(n)], dtype=float)
    return decisions, weights


def _weighted_sign_votes(decisions, weights):

    if decisions.size == 0:
        return np.zeros((decisions.shape[1] if decisions.ndim == 2 else 0,), dtype=float), np.zeros((decisions.shape[1] if decisions.ndim == 2 else 0,), dtype=int)
    signed = 2 * decisions - 1  # 0->-1, 1->+1
    s = np.dot(weights, signed)  # (n,)
    pred = (s > 0).astype(int)
    return s, pred


def _agreement_matrix(decisions):

    m, n = decisions.shape
    if m <= 1:
        return np.zeros(m)
    agree = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(i+1, m):
            agree_ij = np.mean(decisions[i] == decisions[j]) if n > 0 else 1.0
            agree[i, j] = agree_ij
            agree[j, i] = agree_ij
    avg_agree = (agree.sum(axis=1) - np.diag(agree)) / (m - 1)
    return avg_agree


def prune_similar_agents(leader, val_users, val_labels, min_team_size=2, lambda_red=0.3,
                         red_threshold=0.98, score_threshold=-1e-6, max_prune_per_call=3):

    if len(getattr(leader, 'subordinate_agents', [])) <= min_team_size:
        return 0

    f1_before = validate_leader_structure(
        [leader], val_users, val_labels, base_agents=getattr(leader, 'global_base_agents', None)
    )

    removed = 0
    while len(leader.subordinate_agents) > min_team_size and removed < max_prune_per_call:
        decisions, weights = _collect_decision_matrix_and_weights(leader, val_users)
        m = decisions.shape[0]  # —— 使用“当前”团队规模，避免越界 —— #
        if m <= min_team_size or decisions.size == 0:
            break

        y = np.array([val_labels[u] for u in val_users], dtype=int)

        s, pred_with = _weighted_sign_votes(decisions, weights)
        correct_with = (pred_with == y).astype(int)

        signed = 2 * decisions - 1  # (m,n)
        contrib_list = []
        for i in range(m):
            s_wo = s - weights[i] * signed[i]
            pred_wo = (s_wo > 0).astype(int)
            correct_wo = (pred_wo == y).astype(int)
            contrib = float(np.mean(correct_with - correct_wo))
            contrib_list.append(contrib)
        contrib_arr = np.array(contrib_list, dtype=float)

        red_arr = _agreement_matrix(decisions)  

        score = contrib_arr - lambda_red * red_arr

        candidates = [i for i in range(m) if (red_arr[i] >= red_threshold and score[i] <= score_threshold)]
        if not candidates:
            break
        worst = int(sorted(candidates, key=lambda i: score[i])[0])

        old_subs = leader.subordinate_agents[:]
        old_conf = leader.confidence.copy()

        leader.subordinate_agents = [a for idx, a in enumerate(old_subs) if idx != worst]
        new_conf = {}
        new_idx = 0
        for idx in range(len(old_subs)):
            if idx == worst:
                continue
            new_conf[new_idx] = old_conf.get(idx, 1.0)
            new_idx += 1
        leader.confidence = new_conf

        f1_after = validate_leader_structure(
            [leader], val_users, val_labels, base_agents=getattr(leader, 'global_base_agents', None)
        )

        if f1_after + 1e-9 >= f1_before:
            f1_before = f1_after
            removed += 1
        else:
            # 回滚
            leader.subordinate_agents = old_subs
            leader.confidence = old_conf
            break

    return removed



def widen_leader_agent(leader, all_leaders, val_users, val_labels, init_conf=1.0, max_add_per_call=3, margin_eps=0.05, gamma_div=0.2):

    # 基础检查
    existing = getattr(leader, 'subordinate_agents', [])
    if leader is None or len(val_users) == 0:
        return 0

    # 当前团队的验证表现与样本难度
    dec_mat, w = _collect_decision_matrix_and_weights(leader, val_users)
    y = np.array([val_labels[u] for u in val_users], dtype=int)
    s, pred_with = _weighted_sign_votes(dec_mat, w)
    correct_mask = (pred_with == y)
    hard_mask = (~correct_mask) | (np.abs(s) <= margin_eps)
    if hard_mask.sum() == 0:
        return 0  # 没有难样本就不增员

    # 候选：全局 BaseAgents 中不在当前团队的
    global_pool = getattr(leader, 'global_base_agents', []) or []
    team_set = set(id(a) for a in existing)
    candidates = [a for a in global_pool if id(a) not in team_set]

    if not candidates:
        return 0

    # 预取候选的决策行
    def _candidate_row(agent):
        udm = getattr(agent, 'user_decision_map', {})
        return np.array([udm[u]["Decision"] if u in udm else 1 for u in val_users], dtype=int)

    cand_rows = []
    for ag in candidates:
        try:
            cand_rows.append(_candidate_row(ag))
        except Exception:
            cand_rows.append(np.ones(len(val_users), dtype=int))
    cand_rows = np.array(cand_rows, dtype=int)  # (k, n)

    def _max_similarity_to_team(row):
        if dec_mat.size == 0:
            return 0.0
        sims = [np.mean(row == dec_mat[i]) for i in range(dec_mat.shape[0])]
        return max(sims) if sims else 0.0

    signed_team = 2 * dec_mat - 1  # (m,n)
    s_base = s  # (n,)
    gains = []
    div_pen = []
    for k_idx in range(cand_rows.shape[0]):
        row = cand_rows[k_idx]
        s_new = s_base + init_conf * (2 * row - 1)
        pred_new = (s_new > 0).astype(int)
        new_correct = (pred_new == y).astype(int)
        old_correct = (pred_with == y).astype(int)
        gain = float(np.mean((new_correct - old_correct)[hard_mask]))
        gains.append(gain)
        div_pen.append(_max_similarity_to_team(row))
    gains = np.array(gains)
    div_pen = np.array(div_pen)

    added = 0
    old_subs = leader.subordinate_agents[:]
    old_conf = leader.confidence.copy()
    f1_ref = validate_leader_structure([leader], val_users, val_labels, base_agents=getattr(leader, 'global_base_agents', None))

    selected = set()
    for _ in range(max_add_per_call):
        scores = gains - gamma_div * div_pen

        valid_idx = [i for i in range(len(scores)) if i not in selected and scores[i] > 0]
        if not valid_idx:
            break
        best_i = max(valid_idx, key=lambda i: scores[i])

        new_agent = candidates[best_i]
        leader.subordinate_agents.append(new_agent)

        next_idx = len(leader.subordinate_agents) - 1
        leader.confidence[next_idx] = float(init_conf)

        f1_new = validate_leader_structure([leader], val_users, val_labels, base_agents=getattr(leader, 'global_base_agents', None))
        if f1_new + 1e-9 >= f1_ref:

            f1_ref = f1_new
            added += 1
            selected.add(best_i)

            new_row = cand_rows[best_i:best_i+1, :]
            if dec_mat.size == 0:
                dec_mat = new_row.copy()
            else:
                dec_mat = np.vstack([dec_mat, new_row])
        else:
            leader.subordinate_agents.pop()
            leader.confidence.pop(next_idx, None)

            selected.add(best_i)

    if added == 0:
        leader.subordinate_agents = old_subs
        leader.confidence = old_conf

    return added


def get_agent_decision(agent, u):
    if hasattr(agent, "user_decision_map") and agent.user_decision_map:
        return agent.user_decision_map.get(u, {"Decision": 1})["Decision"]

    elif hasattr(agent, "subordinate_agents") and agent.subordinate_agents:
        votes = []
        weights = []
        for j, sub in enumerate(agent.subordinate_agents):
            dec = get_agent_decision(sub, u)
            w = agent.confidence.get(j, 1.0)
            votes.append(dec)
            weights.append(w)
        return int(np.average(votes, weights=weights) > 0.5)

    elif hasattr(agent, "memory") and u in agent.memory:
        return agent.memory[u].get("decision", 1)
    else:
        print(f"[get_agent_decision] Agent {agent.agent_id} has no decision for user {u}.")
        return 1 

def validate_leader_structure(leader_agents, val_users, val_labels, base_agents=None):
    y_true, y_pred = [], []
    for leader in leader_agents:
        for u in val_users:
            pred = get_agent_decision(leader, u)
            y_pred.append(pred)
            y_true.append(val_labels[u])
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return f1

def construct_adaptive_teams_with_reuse(
    base_agents, users, labels, n_groups, all_metrics, client=None, min_group_size=2, max_group_size=None,
    layer_idx=1, team_prefix="", tree_records=None, previous_leader_agents=None, global_base_agents=None
):

    from sklearn.cluster import AgglomerativeClustering
    import numpy as np
    if previous_leader_agents is None:
        previous_leader_agents = []

    def match_old_leader(new_members, old_leader_list):
        # 用 agent_id (= name) 复用
        new_set = set(a.agent_id for a in new_members)
        for old_leader in old_leader_list:
            old_set = set(a.agent_id for a in getattr(old_leader, "subordinate_agents", []))
            if new_set == old_set:
                return old_leader
        return None

    def make_team_name(layer_idx, g, parent_name=None):
        if parent_name:
            return f"{parent_name}-L{layer_idx}-G{g}"
        else:
            return f"L{layer_idx}-G{g}"

    n_agents = len(base_agents)
    dec_mat = get_baseagent_decision_matrix(base_agents, users)  # shape = (n_agents, n_users)
    clustering = AgglomerativeClustering(n_clusters=n_groups, linkage='ward')
    group_labels = clustering.fit_predict(dec_mat)
    group_indices = [[] for _ in range(n_groups)]
    for idx, g in enumerate(group_labels):
        group_indices[g].append(idx)

    leader_agents = []
    for g, idx_list in enumerate(group_indices):
        # 选择分组内F1最优的BaseAgent作为“种子”
        best_f1, leader_idx = -1, idx_list[0]
        for i in idx_list:
            y_true = [labels[u] for u in users if u in base_agents[i].memory]
            y_pred = [base_agents[i].memory[u]['decision'] for u in users if u in base_agents[i].memory]
            if y_true:
                f1 = f1_score(y_true, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    leader_idx = i
        # 构造团队成员，按顺序可复现
        team_indices = sorted(idx_list)
        agents_in_team = [base_agents[i] for i in team_indices]
        # 扩团队以保证 min_group_size
        candidates = [i for i in range(n_agents) if i not in team_indices]
        while len(agents_in_team) < max(len(idx_list), min_group_size):
            best_cand, max_div = None, -1
            for c in candidates:
                div = np.mean([np.sum(dec_mat[c] != dec_mat[member]) for member in team_indices])
                if div > max_div:
                    max_div = div
                    best_cand = c
            if best_cand is not None:
                agents_in_team.append(base_agents[best_cand])
                team_indices.append(best_cand)
                candidates.remove(best_cand)
            else:
                break
        if max_group_size is not None:
            agents_in_team = agents_in_team[:max_group_size]
            team_indices = team_indices[:max_group_size]
        # 构造唯一 team_name
        parent_name = team_prefix if team_prefix else None
        team_name = make_team_name(layer_idx, g, parent_name)
        # 复用老leader
        old_leader = match_old_leader(agents_in_team, previous_leader_agents)
        if old_leader is not None:
            leader_agents.append(old_leader)
        else:
            leader_agent = LeaderAgent(
                metrics=all_metrics,
                client=client,
                subordinate_agents=agents_in_team,
                agent_name=team_name,
                global_base_agents=global_base_agents    # 不要再加 "Leader-" 前缀！
            )
            # 只用名字，不要任何 uuid/随机 id
            leader_agents.append(leader_agent)
    return leader_agents


def recursive_adaptive_structure_with_reuse(
    agents, users, labels, all_metrics, client,
    group_counts, min_group_size=2, max_group_size=None, max_layers=3, layer=1,
    all_leaders=None, tree_records=None, prev_leaders=None, global_base_agents=None
):

    if all_leaders is None:
        all_leaders = []
    if tree_records is None:
        tree_records = []
    if prev_leaders is None:
        prev_leaders = []

    if layer > max_layers or len(agents) <= 1 or layer > len(group_counts):
        return agents, all_leaders, tree_records
    n_groups = group_counts[layer-1]
    leader_agents = construct_adaptive_teams_with_reuse(
        base_agents=agents, users=users, labels=labels, n_groups=n_groups, all_metrics=all_metrics,
        client=client, min_group_size=min_group_size, max_group_size=max_group_size,
        layer_idx=layer, tree_records=tree_records, previous_leader_agents=prev_leaders, global_base_agents=global_base_agents
    )
    all_leaders.extend(leader_agents)
    return recursive_adaptive_structure_with_reuse(
        leader_agents, users, labels, all_metrics, client,
        group_counts, min_group_size, max_group_size, max_layers, layer+1, all_leaders, tree_records, leader_agents, global_base_agents
    )

def get_baseagent_decision_matrix(base_agents, users):

    decision_matrix = []
    for agent in base_agents:
        row = []
        for u in users:
            if u in agent.memory:
                row.append(agent.memory[u]['decision'])
            else:
                row.append(1)
        decision_matrix.append(row)
    return np.array(decision_matrix)

def compute_agent_metrics(agent, users, labels):
    # 优先用 user_decision_map
    if hasattr(agent, 'user_decision_map'):
        y_true = [labels[u] for u in users if u in agent.user_decision_map]
        y_pred = [agent.user_decision_map[u]['Decision'] for u in users if u in agent.user_decision_map]
    else:
        y_true = [labels[u] for u in users if u in agent.memory]
        # 兼容 decision/Decision 字段名
        y_pred = [
            agent.memory[u]['decision']
            if 'decision' in agent.memory[u] else agent.memory[u].get('Decision', 1)
            for u in users if u in agent.memory
        ]
    if not y_true:
        return 0, 0, 0, 0, 0
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except Exception:
        auc = 0
    return accuracy, precision, recall, f1, auc


def save_training_state(
    state_path,
    epoch,
    batch_idx,
    base_agents,
    leader_agents,
    decision_agent,
    user_status=None,
    config_dict=None
):
    # 保存所有BaseAgent
    base_state = {}
    for agent in base_agents:
        base_state[agent.agent_id] = {
            "name": getattr(agent, "name", None),
            "memory": agent.memory,
            "experience": agent.experience,
            "agent_id": agent.agent_id,
        }

    leader_state = {}
    for agent in leader_agents:
        leader_state[agent.agent_id] = {
            "agent_name": getattr(agent, "agent_name", None),
            "memory": agent.memory,
            "experience": getattr(agent, "experience", ""),
            "confidence": getattr(agent, "confidence", {}),
            "agent_id": agent.agent_id,
            "subordinate_ids": [getattr(sub, "agent_id", None) for sub in getattr(agent, "subordinate_agents", [])],
        }

    # 保存决策层
    decision_state = {
        "agent_id": getattr(decision_agent, "agent_id", None),
        "memory": getattr(decision_agent, "memory", {}),
        "experience": getattr(decision_agent, "experience", ""),
        "confidence": getattr(decision_agent, "confidence", {}),
        "subordinate_ids": [getattr(sub, "agent_id", None) for sub in getattr(decision_agent, "subordinate_agents", [])],
        "base_agent_ids": [getattr(sub, "agent_id", None) for sub in getattr(decision_agent, "base_agents", [])],
    }

    state = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "base_agents": base_state,
        "leader_agents": leader_state,
        "decision_agent": decision_state,
        "user_status": user_status or {},
        "config_dict": config_dict or {}
    }
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def load_training_state(state_path, base_agents, leader_agents, decision_agent):
    import os
    if not os.path.exists(state_path):
        return 0, 0, {}, {}

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    # 加载BaseAgent
    base_state = state.get("base_agents", {})
    base_agent_id_map = {agent.agent_id: agent for agent in base_agents}
    for aid, agent_state in base_state.items():
        agent = base_agent_id_map.get(aid, None)
        if agent is not None:
            agent.memory = agent_state.get("memory", {})
            agent.experience = agent_state.get("experience", "")
        else:
            print(f"[load_training_state][BaseAgent] Warning: Agent id {aid} not found in current structure.")

    # 加载LeaderAgent
    leader_state = state.get("leader_agents", {})
    leader_agent_id_map = {agent.agent_id: agent for agent in leader_agents}
    for aid, agent_state in leader_state.items():
        agent = leader_agent_id_map.get(aid, None)
        if agent is not None:
            agent.memory = agent_state.get("memory", {})
            agent.experience = agent_state.get("experience", "")
            agent.confidence = agent_state.get("confidence", {i: 1.0 for i in range(len(getattr(agent, "subordinate_agents", [])))})
        else:
            print(f"[load_training_state][LeaderAgent] Warning: Agent id {aid} not found in current structure.")

    # 加载DecisionAgent
    decision_state = state.get("decision_agent", {})
    if decision_state:
        if getattr(decision_agent, "agent_id", None) == decision_state.get("agent_id", None):
            decision_agent.memory = decision_state.get("memory", {})
            decision_agent.experience = decision_state.get("experience", "")
            decision_agent.confidence = decision_state.get("confidence", {i: 1.0 for i in range(len(getattr(decision_agent, "subordinate_agents", [])))})
        else:
            print("[load_training_state][DecisionAgent] Warning: agent_id mismatch. Not restoring.")

    user_status = state.get("user_status", {})
    config_dict = state.get("config_dict", {})

    return state.get("epoch", 0), state.get("batch_idx", 0), user_status, config_dict


def generate_experience_db_from_text(buffer_path, tokenizer, model, db_path):
    experiences = []
    vectors = []
    with open(buffer_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            exp = entry["experience"]
            # exp 可能是一个list，也可能是str
            # 若是list，则每条都单独存入
            if isinstance(exp, list):
                for e in exp:
                    embedding = get_text_embedding_llama(e, tokenizer, model)
                    experiences.append(e)
                    vectors.append(embedding)
            else:
                embedding = get_text_embedding_llama(exp, tokenizer, model)
                experiences.append(exp)
                vectors.append(embedding)
    save_experience_db(vectors, experiences, db_path)

def save_experience_progress(epoch, batch, path="experience_progress.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"last_epoch": epoch, "last_batch": batch}, f)

def load_experience_progress(total_epochs, total_batches, path="experience_progress.json"):
    if not os.path.exists(path):
        return 0, 0
    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)
    print("total_batches", total_batches)
    last_epoch = state.get("last_epoch", 0)
    last_batch = state.get("last_batch", 0)
    if last_epoch >= total_epochs - 1 and last_batch >= total_batches - 1:
        return "done"
    else:
        return last_epoch, last_batch

def save_experience_text(epoch, batch, users, metrics, experience_text, buffer_path="experience_text_buffer.jsonl"):
    entry = {
        "epoch": epoch,
        "batch": batch,
        "users": users,
        "metrics": metrics,
        "experience": experience_text
    }
    with open(buffer_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def save_experience_db(experience_vectors, experiences, db_path):
    with open(db_path, 'wb') as f:
        pickle.dump({"vectors": experience_vectors, "texts": experiences}, f)

def load_experience_db(db_path):
    with open(db_path, 'rb') as f:
        db = pickle.load(f)
    return db['vectors'], db['texts']

def align_batch_result_with_user_id(user_batch, results):
    # results 是 [{'user_id': xxx, ...}, ...]
    user2result = {r.get("user_id", None): r for r in results if "user_id" in r}
    aligned = []
    for u in user_batch:
        if u in user2result:
            aligned.append(user2result[u])
        else:
            aligned.append({"user_id": u, "Decision": 1, "Reason": "No output"})
    return aligned

def robust_json_parse(content):
    # 去除markdown标记
    content = content.strip()
    content = re.sub(r"^json", "", content)
    content = re.sub(r"$", "", content)
    # 先尝试直接json
    try:
        return json.loads(content)
    except Exception:
        pass
    # 用正则抓取每一个user对象，逐个解析
    obj_pat = r'\{[^{}]*"user_id"[^{}]*\}'
    objs = []
    for m in re.finditer(obj_pat, content):
        txt = m.group()
        # 修补常见错误：Reason: 改成 Reason":
        txt = re.sub(r'"Reason: "', r'"Reason": "', txt)
        # 修补键没引号（可选，如果模型输出全错时才考虑）
        txt = re.sub(r'(\s*)(user_id|Decision|Reason)(\s*):', r'"\2":', txt)
        try:
            objs.append(json.loads(txt))
        except Exception as e:
            print(f"[robust_json_parse] single obj parse failed: {txt}\nerror: {e}")
            continue
    if objs:
        return objs
    print(f"[robust_json_parse] fallback failed: content =\n{content}")
    return []

def safe_json_parse(content):
    content = content.strip()

    # 1. 直接用正则抽第一对[]，无论前面有没有别的内容
    match = re.search(r'\[\s*{.*?}\s*\]', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception as e:
            print("json.loads failed on matched []:", e)
    
    # 2. 如果是单条对象，不是list，也解析一下
    match2 = re.search(r'\{\s*"user_id"\s*:.*?\}', content, re.DOTALL)
    if match2:
        try:
            return [json.loads(match2.group())]
        except Exception as e:
            print("json.loads failed on matched single obj:", e)

    # 3. 如果前缀有 "json" 或 markdown 标记
    if content.startswith("json"):
        content = content[4:].strip()
    if content.startswith("```json"):
        content = content[7:].strip()
    if content.endswith("```"):
        content = content[:-3].strip()
    # 再尝试直接解析
    try:
        return json.loads(content)
    except Exception as e:
        pass

    # 4. 尝试补右中括号（list不完整的情况）
    if content.startswith("[") and not content.endswith("]"):
        try:
            fix_content = content + "]"
            return json.loads(fix_content)
        except Exception:
            pass

    # 5. 逐项抓取对象，组装成list
    objs = []
    pattern = r'{.*?}(?=,|$|\n|\r)'
    for m in re.finditer(pattern, content, flags=re.DOTALL):
        txt = m.group()
        try:
            objs.append(json.loads(txt))
        except Exception:
            continue
    if objs:
        return objs

    # 6. fallback 最后再用一次通用[]抽取
    match = re.search(r'\[.*\]', content, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    # 7. 新增：尝试单个dict（非list，非user_id），比如"{"correct_experience":...}"
    match3 = re.search(r'\{[\s\S]*\}', content)
    if match3:
        try:
            return json.loads(match3.group())
        except Exception:
            pass

    print(f"safe_json_parse fallback failed: content truncated or broken")
    print("content", repr(content))  # repr更安全
    return []


def clean_json_prefix(content):
    content = content.strip()
    if content.startswith("```json"):
        content = content[len("```json"):].strip()
    if content.startswith("```"):
        content = content[len("```"):].strip()
    if content.endswith("```"):
        content = content[:-3].strip()
    if content.startswith("json"):
        content = content[4:].strip()
    return content
            
def call_llm_until_response(prompt, question, client, max_retry=5, sleep_time=2):
    """循环调用LLM直到获得非空结果"""
    for attempt in range(max_retry):
        try:
            text = agent_semantic(prompt, question, client)
            # print(f"[DEBUG][LLM] attempt {attempt} return: {text}")
            if text and isinstance(text, str) and len(text.strip()) > 0:
                print("text", text)
                return text
        except Exception as e:
            print(f"[DEBUG][LLM] attempt {attempt} return: {text}")
            print(f"[API ERROR] {e}, retrying...")
        time.sleep(sleep_time)
    print("[API WARNING] Max retries reached, returning empty string.")
    return ""

def agent_semantic(prompt, question, client):
    response = client.chat.asyncCompletions.create(
        model="glm-4-FlashX-250414",  # 请根据实际情况指定模型名称
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ],
        max_tokens=500,  # 限制输出内容在200 tokens以内
        temperature=0.05,  # 降低随机性，增强确定性
        top_p=0.95 # 限制采样概率，增强确定性
    )
    task_id = response.id
    task_status = ''
    get_cnt = 0
    while task_status not in ['SUCCESS', 'FAILED'] and get_cnt <= 40:
        result_response = client.chat.asyncCompletions.retrieve_completion_result(id=task_id)
        task_status = result_response.task_status
        time.sleep(2)
        get_cnt += 1
    content = result_response.choices[0].message.content
    return " ".join(content.split())

def parallel_reasoning(agents, user_id, user_metrics, agent_level, agent_idx_list, **kwargs):
    # 支持L3、L2（可通过传入subordinate_results等参数实现更灵活聚合）
    results = [None] * len(agents)
    def reasoning_task(agent, idx):
        if 'subordinate_results' in kwargs and kwargs['subordinate_results'] is not None:
            # L2/L1
            decision, reason = agent.reasoning(user_id, user_metrics, kwargs['subordinate_results'][idx], agent_level=agent_level, agent_idx=idx)
            return {'decision': decision, 'reason': reason}
        else:
            decision, reason = agent.reasoning(user_id, user_metrics, agent_level=agent_level, agent_idx=idx)
            return {'decision': decision, 'reason': reason}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(agents)) as executor:
        futures = {executor.submit(reasoning_task, agent, idx): idx for idx, agent in zip(agent_idx_list, agents)}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    return results

def parallel_refinement(agents, user_id, user_metrics, ground_truth, decisions, reasons, feedbacks, agent_level, agent_idx_list):
    def refinement_task(agent, idx, decision, reason, feedback):
        agent.refine(user_id, user_metrics, ground_truth, decision, reason, feedback, agent_level=agent_level, agent_idx=idx)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(agents)) as executor:
        futures = [
                    executor.submit(refinement_task, agent, idx, decision, reason, feedback)
                    for agent, idx, decision, reason, feedback in zip(agents, l3_needs_refine, decisions, reasons, feedbacks)
                ]
        concurrent.futures.wait(futures)

default_cache_path = 'llama_user_embedding_cache.pkl'

def agent_log(event, agent_level, agent_idx, user_id, step, input_dict, output_dict):
    print(f"[{event}] [Level:{agent_level}][Idx:{agent_idx}][User:{user_id}][{step}]")
    print("  Input:")
    for k, v in input_dict.items():
        print(f"    {k}: {str(v)}")
    print("  Output:")
    for k, v in output_dict.items():
        print(f"    {k}: {str(v)}")
    print("-" * 80)

def clean_generated_text(text):
    """清洗生成的文本，去除多余空格和换行，返回单行文本。"""
    return " ".join(text.split())

def safe_usage_get(usage, key, default=0):
    if usage is None:
        return default
    if isinstance(usage, dict):
        return usage.get(key, default)
    # 兼容Pydantic对象
    try:
        return getattr(usage, key)
    except Exception:
        return default

def agent_reasoning(prompt, question, client):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ],
        max_tokens=5000,
        temperature=0.1,
        top_p=0.95
    )
    content = response.choices[0].message.content
    usage = getattr(response, 'usage', None)
    input_tokens = safe_usage_get(usage, "prompt_tokens", 0)
    output_tokens = safe_usage_get(usage, "completion_tokens", 0)
    return " ".join(content.split()), input_tokens, output_tokens

def agent_refine(prompt, question, client):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ],
        max_tokens=5000,
        temperature=0.1,
        top_p=0.95
    )
    content = response.choices[0].message.content
    usage = getattr(response, 'usage', None)
    input_tokens = safe_usage_get(usage, "prompt_tokens", 0)
    output_tokens = safe_usage_get(usage, "completion_tokens", 0)
    return " ".join(content.split()), input_tokens, output_tokens
    

# --------------------------- Data Loading ---------------------------
def load_item_info(filepath):
    item_info = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                item_id, title = parts[0], parts[1]
                item_info[item_id] = title
    return item_info

def load_simulated_dataset(filepath, item_info):
    dataset = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 5:
                user = parts[0]
                interaction = {
                    "item_id": parts[1],
                    "rating": float(parts[2]),
                    "review_body": parts[3],
                    "timestamp": parts[4]
                }
                dataset.setdefault(user, []).append(interaction)
    return dataset

def load_user_labels(filepath):
    users, labels = [], {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                user, label = parts[0], int(parts[1])
                users.append(user)
                labels[user] = label
    return users, labels

# --------------------------- Agent Classes ---------------------------
class BaseAgent:
    _id_counter = 0
    def __init__(self, metrics, client=None, name=None):
        self.profile = metrics      # metrics key list
        self.memory = {}
        self.experience = ""
        self.client = client        # 每个agent都可有自己的client实例
        if name:
            self.name = name
        else:
            self.name = f"BaseAgent-{BaseAgent._id_counter}"
            BaseAgent._id_counter += 1
        self.agent_id = self.name

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ['client', 'model', 'tokenizer']:
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def batch_reasoning(self, user_ids, user_metrics_list, agent_level='L3', agent_idx=None, step="reasoning"):

        prompt = (
            "You are one member of detection expert team analyzing multiple users' metrics to determine if each user is malicious (1) or normal (0).\n"
            "For each user, output your decision and corresponding reason in JSON list format:\n"
            "Format Example:\n"
            '[\n  {"user_id": "User1", "Decision": 0, "Reason": "..."},\n  {"user_id": "User2", "Decision": 1, "Reason": "..."}\n]\n'
            "Ensure the JSON is valid, complete, syntactically correct, not truncated, and output nothing else.\n"
        )
        questions = []
        for user_id, user_metrics in zip(user_ids, user_metrics_list):
            metrics_input = {k: user_metrics[k] for k in self.profile}
            q = (
                f"User ID: {user_id}\n"
                f"Metrics: {metrics_input}\n"
                f"Experience: {self.experience}\n"
            )
            questions.append(q)
        questions_list = "\n\n".join(questions)
        print("questions_list", questions_list)
        content, input_tokens, output_tokens = agent_reasoning(prompt, questions_list, self.client)
        # 解析json
        try:
            results = safe_json_parse(content)
        except Exception as e:
            print(f"[BaseAgent.batch_reasoning] JSONDecodeError: {e}, content=\n{content}")
            results = []
        return results, input_tokens, output_tokens

    def batch_refine(self, user_ids, user_metrics_list, ground_truths, decisions, reasons, feedbacks,
                    agent_level='L3', agent_idx=None, step="refine"):
        prompt = (
            "You are a detection expert. For the following users, your previous decisions were incorrect. "
            "Please summarize, in one concise sentence, what you learned from these errors and leader feedback, "
            "and how you will update your experience to improve future reasoning.\n"
            "Return your answer in this strict JSON format:\n"
            '{"New Experience": "<your updated experience in one sentence, not user-specific>"}'
            "Ensure the JSON is valid, complete, syntactically correct, not truncated, and output nothing else.\n"
        )
        questions = []
        for user_id, user_metrics, gt, dec, reason, feedback in zip(user_ids, user_metrics_list, ground_truths, decisions, reasons, feedbacks):
            metrics_input = {k: user_metrics[k] for k in self.profile}
            q = (
                f"User ID: {user_id}\n"
                f"Metrics: {metrics_input}\n"
                f"Ground Truth: {gt}\n"
                f"Previous Decision: {dec}\n"
                f"Previous Reasoning: {reason}\n"
                f"Previous Experience: {self.experience}\n"
                f"Leader Feedback: {feedback}\n"
            )
            questions.append(q)
        content, input_tokens, output_tokens = agent_refine(prompt, "\n\n".join(questions), self.client)
        try:
            content = content.strip()
            if content.startswith("json"): content = content[7:]
            if content.endswith(""): content = content[:-3]
            result_json = json.loads(content)
            self.experience = result_json.get("New Experience", self.experience)
        except Exception as e:
            print(f"[BaseAgent.batch_refine] JSONDecodeError: {e}, content=\n{content}")
        return self.experience, input_tokens, output_tokens

class ImprovedBaseAgent(BaseAgent):
    def __init__(self, metrics, client=None, tokenizer=None, model=None, experience_db_path='experience_db.pkl',
                 user_list=None, user_metrics_dict=None, precomputed_decision_path=None, k=20, name="None"):
        super().__init__(metrics, client)
        self.tokenizer = tokenizer
        self.model = model
        self.experience_db_path = experience_db_path
        self.user_list = user_list
        self.user_metrics_dict = user_metrics_dict
        self.precomputed_decision_path = precomputed_decision_path or "precomputed_improved_agent_results.json"
        self.k = k
        self.user_decision_map = {}
        self.experience_file = f"experience_{name}.json"
        self.experience = self.load_or_generate_experience()
        # 初始化用户决策缓存
        self._init_decision_cache()
        self.name = name  # 新增：agent名字
        all_users = self.user_list
        print("[ImprovedBaseAgent] Initialized:", self.name)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ['client', 'model', 'tokenizer']:
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def load_or_generate_experience(self):
        # 优先加载本地experience文件
        if os.path.exists(self.experience_file):
            with open(self.experience_file, "r", encoding="utf-8") as f:
                exp = json.load(f).get("experience", "")
                if exp:
                    return exp
        # 否则自动采样并持久化
        exp = self.initialize_experience_and_decisions()
        with open(self.experience_file, "w", encoding="utf-8") as f:
            json.dump({"experience": exp}, f, ensure_ascii=False, indent=2)
        return exp

    def initialize_experience_and_decisions(self, k=None):
        # 采样经验，原接口
        vectors, exps = load_experience_db(self.experience_db_path)
        k = k or self.k
        idxs = random.sample(range(len(exps)), min(k, len(exps)))
        sampled = [exps[i] for i in idxs]
        return self.summarize_experiences(sampled)

    def summarize_experiences(self, experiences):
        # 与GlobalExperienceAgent一致
        prompt = (
            "You are an expert agent initializing your experience for a set of metrics in malicious user detection. "
            "The following are several expert experience summaries. Based on these, generate your own concise and general experience. "
            "Output format: {\"experience\": \"<your summarized experience>\"}"
        )
        question = "\n".join(experiences)
        summary_json, _, _ = agent_refine(prompt, question, self.client)
        summary_json = summary_json.strip() if summary_json else ""
        if not summary_json:
            print("[summarize_experiences] Empty response from LLM, use default experience!")
            return "No valid experience generated."
        try:
            summary_dict = json.loads(summary_json)
            return summary_dict["experience"]
        except Exception as e:
            print(f"[summarize_experiences] JSONDecodeError: {e}, summary_json=\n{summary_json}")
            return "No valid experience generated."

    def _init_decision_cache(self):
        # 不改变原有决策缓存流程
        missing_users = []
        if os.path.exists(self.precomputed_decision_path):
            with open(self.precomputed_decision_path, "r", encoding="utf-8") as f:
                self.user_decision_map = json.load(f)
            missing_users = [u for u in self.user_list if u not in self.user_decision_map]
        else:
            missing_users = list(self.user_list)
        if missing_users:
            batch_size = 10
            for i in range(0, len(missing_users), batch_size):
                batch = missing_users[i:i+batch_size]
                batch_metrics = [self.user_metrics_dict[u] for u in batch]
                # 如果你原本就是缓存推理结果，这里可以用真实推理逻辑
                results, _, _ = super().batch_reasoning(batch, batch_metrics)
                print("results", results)
                for u, res in zip(batch, results):
                    if isinstance(res, dict) and "Decision" in res and "Reason" in res:
                        self.user_decision_map[u] = {"Decision": res["Decision"], "Reason": res["Reason"]}
                    else:
                        self.user_decision_map[u] = {"Decision": 1, "Reason": "No valid output."}
                with open(self.precomputed_decision_path, "w", encoding="utf-8") as f:
                    json.dump(self.user_decision_map, f, ensure_ascii=False, indent=2)
            with open(self.precomputed_decision_path, "w", encoding="utf-8") as f:
                json.dump(self.user_decision_map, f, ensure_ascii=False, indent=2)

    def batch_reasoning(self, user_ids, user_metrics_list, **kwargs):
        results = []
        for u in user_ids:
            r = self.user_decision_map.get(u, {"Decision": 1, "Reason": "No cached decision."})
            # 新增：同步存储到 memory
            self.memory[u] = {
                "decision": r["Decision"],
                "reason": r["Reason"],
                # 可以补充更多信息（如 metrics、step 等）
            }
            results.append({"user_id": u, "Decision": r["Decision"], "Reason": r["Reason"]})
        return results, 0, 0

    def batch_refine(self, *args, **kwargs):
        return self.experience, 0, 0

# Batched LeaderAgent
class LeaderAgent:
    _id_counter = 0
    def __init__(self, metrics, client=None, subordinate_agents=None, agent_name=None, global_base_agents=None):
        self.profile = metrics
        self.memory = {}
        self.client = client
        self.subordinate_agents = subordinate_agents or []
        if agent_name:
            self.agent_name = agent_name
        else:
            self.agent_name = f"Leader-{LeaderAgent._id_counter}"
            LeaderAgent._id_counter += 1
        self.agent_id = self.agent_name  # 只用 name
        self.confidence = {i: 1.0 for i in range(len(self.subordinate_agents))}
        self.global_base_agents = global_base_agents
        
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ['client', 'model', 'tokenizer']:
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def sync_confidence(self):
        # self.confidence 是 0~len(self.subordinate_agents)-1 映射
        # 没有的下标都补1.0，多余的删除
        n = len(self.subordinate_agents)
        new_conf = {i: self.confidence.get(i, 1.0) for i in range(n)}
        self.confidence = new_conf

    def batch_reasoning(self, user_ids, user_metrics_list, batch_subordinate_results=None, agent_level=None, agent_idx=None, step="reasoning"):
        self.sync_confidence()
        # print("[DEBUG] LeaderAgent输入:", batch_subordinate_results)
        if batch_subordinate_results is None and self.subordinate_agents:
            all_sub_outputs = []
            for agent in self.subordinate_agents:
                sub_results, _, _ = agent.batch_reasoning(user_ids, user_metrics_list)
                all_sub_outputs.append(sub_results)
            batch_subordinate_results = list(map(list, zip(*all_sub_outputs)))  # [batch][num_sub]
        results = []
        for i, u in enumerate(user_ids):
            votes = []
            for j, sub in enumerate(self.subordinate_agents):
                dec = batch_subordinate_results[i][j]['Decision']
                w = self.confidence[j]
                votes.append((dec, w))
            decision = int(np.average([d for d, w in votes], weights=[w for d, w in votes]) > 0.5)
            conf = np.mean([w for d, w in votes])
            # 这里写入memory
            self.memory[u] = {
                "decision": decision,
                "confidence": conf,
                "votes": votes,
                "agent_level": agent_level,
                "step": step
            }
            results.append({"user_id": u, "Decision": decision, "Confidence": conf, "Votes": votes})
        # print("[DEBUG] LeaderAgent输出:", results)
        return results, 0, 0

    def batch_refine(self, user_ids, user_metrics_list, ground_truths, decisions, reasons, feedbacks,
                    batch_subordinate_results, agent_level=None, agent_idx=None, step="refine",
                    val_users=None, val_labels=None):
        """
        LLM生成新confidence后，对比val集F1，仅F1提升时才应用权重更新，否则回滚。
        """
        print(f"[LeaderAgent-{self.agent_name}] batch_refine called.")
        prompt = (
            "You are the team leader in a multi-agent malicious user detection team. "
            "For each batch of users, you must re-evaluate your trust (weight, range 0~1) in each subordinate agent, "
            "based on their decisions, reasons, current weights, and whether their decisions match the ground truth.\n"
            "IMPORTANT: To maximize team effectiveness, avoid assigning all subordinates the same or nearly identical weights. "
            "You MUST reflect the actual contribution, reliability, or diversity of each subordinate by using clearly different weights (e.g., not all 0.5). "
            "However, to ensure the stability of the team, you MUST only make **small incremental adjustments** to each subordinate's weight in each update. "
            "Specifically, for each subordinate, the change from the current weight should be a small value in the range of 0.001 to 0.009 (increase or decrease). "
            "All weights must keep **exactly four digits after the decimal point**.\n"
            "Output a single JSON dict:\n"
            "\"new_weights\": <dict of updated weights for each subordinate (key: index, value: float to 4 decimal places)>.\n"
            "For example:\n"
            "{ \"new_weights\": {\"0\": 0.7312, \"1\": 0.7269, \"2\": 0.7354, \"3\": 0.7407} }\n"
            "If a current weight is 0.7302, the new value must be within [0.7212, 0.7392].\n"
            "Bad Example (not allowed): { \"new_weights\": {\"0\": 0.5, \"1\": 0.5, \"2\": 0.5, \"3\": 0.5} }\n"
            "Only output the JSON, output numbers only (no expressions, no explanations)."
        )


        questions = []
        for user_id, user_metrics, gt, dec, reason, feedback, subordinate_results in zip(
                user_ids, user_metrics_list, ground_truths, decisions, reasons, feedbacks, batch_subordinate_results):
            if dec is None:
                dec = 0
            input_json = {
                "subordinates": [
                    {
                        "index": i,
                        "weight": float(self.confidence.get(i, 1.0)),
                        "decision": int(sub.get("Decision", sub.get("decision", 1))),
                        "reason": str(sub.get("Reason", sub.get("reason", ""))),
                        "match_gt": int(sub.get("Decision", sub.get("decision", 1))) == int(gt)
                    } for i, sub in enumerate(subordinate_results)
                ],
                "leader_final_decision": int(dec),
                "ground_truth": int(gt),
                "leader_reason": reason,
                "feedback": feedback,
            }
            q = f"User ID: {user_id}\nInput: {json.dumps(input_json, indent=2)}"
            questions.append(q)
        content, input_tokens, output_tokens = agent_refine(prompt, "\n\n".join(questions), self.client)

        # ==== 核心部分: 权重更新前保存old_confidence ====
        old_confidence = self.confidence.copy()

        # ==== 解析LLM输出并应用新权重（临时） ====
        try:
            content = content.strip()
            if content.startswith("```json"): content = content[7:].strip()
            if content.endswith("```"): content = content[:-3].strip()
            new_weights_dict = json.loads(content)
            wdict = new_weights_dict.get("new_weights", {})
            for k, v in wdict.items():
                try:
                    new_v = float(v)
                    self.confidence[int(k)] = max(0.0, min(new_v, 1.0))
                except Exception as e:
                    print(f"[batch_refine] skip invalid weight: {k}:{v}")
        except Exception as e:
            print(f"[LeaderAgent.batch_refine] JSONDecodeError: {e}, content=\n{content}")

        # ==== F1对比&回滚机制 ====
        if val_users is not None and val_labels is not None:
            # 先还原旧权重，计算F1
            temp_conf = self.confidence.copy()
            self.confidence = old_confidence
            f1_before = validate_leader_structure([self], val_users, val_labels, base_agents=self.global_base_agents)
            # 再应用新权重，计算F1
            self.confidence = temp_conf
            f1_after = validate_leader_structure([self], val_users, val_labels, base_agents=self.global_base_agents)
            # 对比并决定是否回滚
            if f1_after < f1_before:
                print(f"[LeaderAgent-{self.agent_name}] Refine后F1下降，撤销权重更新。F1 {f1_before:.4f}→{f1_after:.4f}")
                self.confidence = old_confidence
            else:
                print(f"[LeaderAgent-{self.agent_name}] Refine后F1提升或持平，权重更新。F1 {f1_before:.4f}→{f1_after:.4f}")

        return self.confidence, input_tokens, output_tokens

class DecisionAgent(BaseAgent):
    """
    决策层Agent：综合所有用户特征、各LeaderAgent决策与理由（理由从支持该决策的BaseAgent采样），
    支持reasoning和refine，自适应更新confidence，且能总结经验反哺BaseAgent。
    """
    def __init__(self, metrics, client=None, subordinate_agents=None, base_agents=None, tokenizer=None, model=None, k=5, experience_path="experience_decisionagent.json"):
        super().__init__(metrics, client)
        self.subordinate_agents = subordinate_agents or []
        self.base_agents = base_agents or []
        self.tokenizer = tokenizer
        self.model = model
        self.confidence = {i: 1.0 for i in range(len(self.subordinate_agents))}
        self.memory = {}  # user_id: {...}
        self.k = k
        self.experience_path = experience_path

        # ======= 新增部分：优先加载历史经验 =======
        self.experience = load_decision_experience(self.experience_path)
        if not self.experience:
            self.experience = []
            self.experience.append(self.initialize_experience_and_decisions())
            save_decision_experience(self.experience, self.experience_path)
        # =====================================

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ['client', 'model', 'tokenizer']:
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def batch_reasoning_and_summarize(self, user_ids, user_metrics_list, labels, tokenizer, model, client):
        experiences, y_true, y_pred = [], [], []
        for user_id, user_metrics in zip(user_ids, user_metrics_list):
            decision_batch, _, _ = self.batch_reasoning([user_id], [user_metrics])
            decision = decision_batch[0]['Decision']
            reason = decision_batch[0]['Reason']
            ground_truth = labels[user_id]
            feedback = "correct" if decision == ground_truth else "incorrect"
            experience_summary = f"Metrics: {user_metrics}. Decision: {decision}, Reason: {reason}. Result was {feedback}."
            exp_text = self.generate_experience(experience_summary, tokenizer, model, client, user_id)
            experiences.append(exp_text)
            y_true.append(ground_truth)
            y_pred.append(decision)
        recall = recall_score(y_true, y_pred)
        return experiences, recall

    def generate_experience(self, experience_summary, tokenizer, model, client, user_id):
        prompt = (
            "You are summarizing your learning as an expert in malicious user detection. "
            "Based on the following detection process and feedback, summarize a general detection experience in one concise sentence for future reference. Output only the summary sentence."
        )
        question = experience_summary
        cleaned_text, _, _ = agent_refine(prompt, question, client)
        return cleaned_text

    def initialize_experience_and_decisions(self, k=None):
        base_exps = []
        for agent in self.base_agents:
            if hasattr(agent, "experience"):
                base_exps.append(agent.experience)
        if len(base_exps) < (k or self.k):
            base_exps.extend(["Normal behavior", "Anomalous metric patterns", "Benefit of the doubt"] * ((k or self.k) - len(base_exps)))
        sampled = random.sample(base_exps, min(k or self.k, len(base_exps)))
        return self.summarize_experiences(sampled)

    def summarize_experiences(self, experiences):
        prompt = (
            "You are the ultimate decision expert. Given the following detection experience summaries from team members, "
            "summarize one concise detection experience sentence to optimize future team decision-making. "
            "Output format: {\"experience\": \"<your summarized experience>\"}"
        )
        question = "\n".join(experiences)
        summary_json, _, _ = agent_refine(prompt, question, self.client)
        summary_json = summary_json.strip() if summary_json else ""
        if not summary_json:
            return "No valid experience generated."
        try:
            summary_dict = json.loads(summary_json)
            return summary_dict["experience"]
        except Exception:
            return "No valid experience generated."

    def batch_reasoning(self, user_ids, user_metrics_list, batch_subordinate_results=None, agent_level="Decision", agent_idx=None, step="reasoning"):
        # print("[DEBUG] DecisionAgent输入:", batch_subordinate_results)
        results = []
        for i, user_id in enumerate(user_ids):
            metrics_input = user_metrics_list[i]
            leader_votes = []
            leader_reasons = []
            leader_confidences = []
            if batch_subordinate_results is not None:
                for leader_idx, leader_result in enumerate(batch_subordinate_results[i]):
                    leader_dec = leader_result.get("Decision", 1)
                    leader_agent = self.subordinate_agents[leader_idx]
                    chosen_reason = "No supporting reason."
                    leader_conf = self.confidence.get(leader_idx, 1.0)
                    if hasattr(leader_agent, "subordinate_agents"):
                        subs = leader_agent.subordinate_agents
                        base_results = []
                        if hasattr(subs[0], 'user_decision_map'):
                            for subidx, ba in enumerate(subs):
                                sub_res = ba.user_decision_map.get(user_id, {"Decision": 1, "Reason": "No cache."})
                                if sub_res.get("Decision") == leader_dec:
                                    base_results.append(sub_res.get("Reason"))
                        if base_results:
                            chosen_reason = random.choice(base_results)
                    leader_votes.append(leader_dec)
                    leader_reasons.append(chosen_reason)
                    leader_confidences.append(leader_conf)
            else:
                leader_votes = [1] * len(self.subordinate_agents)
                leader_reasons = ["No subordinate result."] * len(self.subordinate_agents)
                leader_confidences = [1.0] * len(self.subordinate_agents)

            prompt = (
                "You are the final decision expert for user classification.\n"
                "Below are the user's full metrics and all team leaders' decisions and their supporting reasons.\n"
                "\n"
                "Decision rules you MUST follow:\n"
                "1) If there is a CONFLICT between high-risk signals in Metrics and the aggregated LeaderVotes (e.g., mlp_score is high or multiple metrics indicate risk, while the majority of LeaderVotes predict 0/benign), you must BIAS TOWARD predicting malicious (Decision=1) unless there is overwhelming counterevidence in the metrics themselves.\n"
                "   - Concrete example: if mlp_score is high but most LeaderVotes are 0, prefer Decision=1 and explicitly state this conflict in Reason.\n"
                "2) Keep your adjustment modest when evidence is weak, but still lean to Decision=1 under the above conflict rule.\n"
                "\n"
                "Output requirement:\n"
                "For each user, output your decision and corresponding reason in JSON list format:\n"
                "[\n  {\"user_id\": \"User1\", \"Decision\": 0, \"Reason\": \"...\"},\n  {\"user_id\": \"User2\", \"Decision\": 1, \"Reason\": \"...\"}\n]\n"
                "Ensure the JSON is valid, complete, syntactically correct, not truncated, and output nothing else.\n"
            )

            question = (
                f"User ID: {user_id}\n"
                f"Metrics: {metrics_input}\n"
                f"LeaderVotes: {leader_votes}\n"
                f"LeaderReasons: {leader_reasons}\n"
                f"LeaderConfidences: {leader_confidences}\n"
                f"Experience: {self.experience}\n"
            )
            resp, _, _ = agent_reasoning(prompt, question, self.client)
            print(f"[DecisionAgent] user_id={user_id}, resp={resp}")
            def here_safe_json_parse(content):
                # 去除markdown代码块和前缀
                content = content.strip()
                # 去掉markdown代码块前后缀
                if content.startswith("```json"):
                    content = content[len("```json"):].strip()
                if content.startswith("```"):
                    content = content[len("```"):].strip()
                if content.endswith("```"):
                    content = content[:-3].strip()
                # 再尝试用正则抓第一个合法的[]
                match = re.search(r'\[\s*{.*?}\s*\]', content, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group())
                    except Exception:
                        pass
                # 直接load
                try:
                    return json.loads(content)
                except Exception:
                    return None
            result = here_safe_json_parse(resp)
            if isinstance(result, list):
                if len(result) > 0 and isinstance(result[0], dict):
                    result = result[0]
                else:
                    result = {"Decision": 1, "Reason": "LLM returned empty or invalid list."}
            elif not isinstance(result, dict):
                result = {"Decision": 1, "Reason": "LLM returned invalid format."}
            if result is None:
                result = {"Decision": 1, "Reason": "LLM format error."}

            self.memory[user_id] = {
                "decision": result.get("Decision", 1),
                "reason": result.get("Reason", "No reason"),
                "metrics": metrics_input,
                "leader_votes": leader_votes,
                "leader_reasons": leader_reasons,
                "is_correct": None,
                "refine_experience": None
            }
            results.append({"user_id": user_id, "Decision": result.get("Decision", 1), "Reason": result.get("Reason", "No reason")})
        print("[DEBUG] DecisionAgent输出:", results)
        return results, 0, 0

    def batch_refine(self, user_ids, user_metrics_list, ground_truths, decisions, reasons, feedbacks, batch_subordinate_results, agent_level="Decision", agent_idx=None, step="refine"):
        prompt = (
            "You are the final decision expert. For each user, your last decision was recorded. "
            "Based on the user's true label, your previous decision, and team information, "
            "summarize: (1) what you learned from correct decisions; (2) what you learned from wrong decisions; "
            "and suggest confidence updates for each subordinate leader (0~1).\n"
            "IMPORTANT: Only make **small incremental adjustments** to each subordinate's weight. "
            "For each subordinate, the change from the current weight should be a small value in the range of 0.005 to 0.05 (increase or decrease). "
            "All weights must have exactly **four digits after the decimal point**.\n"
            "Output: {\"correct_experience\": \"...\", \"error_experience\": \"...\", \"new_weights\": {\"0\":0.8013, \"1\":0.7987, ...}}\n"
            "For example, if the current weight is 0.8002, the new value must be within [0.7912, 0.8092].\n"
            "Only output the JSON, output numbers only (no expressions, no explanations)."
        )

        questions = []
        for i, user_id in enumerate(user_ids):
            input_json = {
                "user_id": user_id,
                "metrics": user_metrics_list[i],
                "ground_truth": ground_truths[i],
                "your_decision": decisions[i],
                "your_reason": reasons[i],
                "leader_votes": [x.get("Decision", 1) for x in batch_subordinate_results[i]],
                "leader_reasons": [x.get("Reason", "") for x in batch_subordinate_results[i]],
                "confidence": self.confidence,
                "feedback": feedbacks[i],
            }
            questions.append(json.dumps(input_json, ensure_ascii=False))
        content, input_tokens, output_tokens = agent_refine(prompt, "\n".join(questions), self.client)
        try:
            content = content.strip()
            if content.startswith("```json"): content = content[7:].strip()
            if content.endswith("```"): content = content[:-3].strip()
            def remove_json_comments(content):
                content = content.strip()
                # 优先找 ```json ... ``` markdown 块
                code_blocks = re.findall(r"```json([\s\S]*?)```", content)
                if code_blocks:
                    json_str = code_blocks[-1].strip()
                else:
                    # 否则找最后一个 { ... }
                    matches = re.findall(r'(\{[\s\S]*\})', content)
                    if not matches:
                        raise ValueError("No valid JSON found in LLM output:\n" + content)
                    json_str = matches[-1].strip()

                def remove_json_comments(text):
                    # 去除行内 // 注释
                    text = re.sub(r'//.*', '', text)
                    # 去除对象和数组里的末尾逗号
                    text = re.sub(r',(\s*[}\]])', r'\1', text)
                    return text

                json_str = remove_json_comments(json_str)
                return json.loads(json_str)
            content = remove_json_comments(content)
            result = json.loads(content)
            if isinstance(content, dict):
                result = content
            else:
                result = json.loads(content)
            correct_exp = result.get("correct_experience", "")
            error_exp = result.get("error_experience", "")
            wdict = result.get("new_weights", {})
            for k, v in wdict.items():
                try:
                    self.confidence[int(k)] = float(v)
                except Exception:
                    pass
            # 补充memory
            for i, user_id in enumerate(user_ids):
                is_correct = (decisions[i] == ground_truths[i])
                if user_id in self.memory:
                    self.memory[user_id]["is_correct"] = is_correct
                    self.memory[user_id]["refine_experience"] = {
                        "correct_exp": correct_exp,
                        "error_exp": error_exp
                    }
        except Exception as e:
            print(f"[DecisionAgent.batch_refine] JSONDecodeError: {e}, content=\n{content}")
        return self.confidence, input_tokens, output_tokens

    def summarize_epoch_experience(self, error_sample_ratio=0.6, correct_sample_ratio=0.2, min_error=5, min_correct=2):
        errors = [v["reason"] for v in self.memory.values() if v.get("is_correct") == False]
        corrects = [v["reason"] for v in self.memory.values() if v.get("is_correct") == True]
        n_error = max(int(len(errors) * error_sample_ratio), min_error) if errors else 0
        n_correct = max(int(len(corrects) * correct_sample_ratio), min_correct) if corrects else 0

        sampled = []
        if errors:
            sampled += random.sample(errors, min(n_error, len(errors)))
        if corrects:
            sampled += random.sample(corrects, min(n_correct, len(corrects)))

        if not sampled:
            return "No valid epoch experience."
        return self.summarize_experiences(sampled)


# --------------------------- Deep Learning Model ---------------------------
def extract_features_llama_with_cache(user_ids, simulated_data, tokenizer, model, cache_path=default_cache_path):
    import pickle
    import os
    # 加载或初始化缓存
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cache_dict = pickle.load(f)
    else:
        cache_dict = {}

    # 计算所有唯一的 item_id（按字典序排序），并排除最后一个项目
    all_items = sorted({interaction["item_id"] 
                        for interactions in simulated_data.values() 
                        for interaction in interactions})
    if len(all_items) < 2:
        raise ValueError("数据中 item 数量不足，无法构造固定长度评分向量。")
    # 构造映射，长度固定为项目数量-1
    item_to_index = {item: idx for idx, item in enumerate(all_items[:-1])}
    fixed_length = len(item_to_index)  # 即 total_items - 1

    features = []
    for user_id in user_ids:
        interactions = simulated_data.get(user_id, [])
        # 构造固定长度评分向量，初始化为0
        rating_vector = np.zeros(fixed_length, dtype=np.float32)
        # 根据该用户对于某 item 的评价，若有多个评分取平均
        item_ratings = {}
        for inter in interactions:
            item = inter["item_id"]
            if item in item_to_index:
                item_ratings.setdefault(item, []).append(inter["rating"])
        for item, ratings in item_ratings.items():
            idx = item_to_index[item]
            # 取平均评分
            rating_vector[idx] = np.mean(ratings)
            
        review_text = " ".join([i["review_body"] for i in interactions])
        text_embedding = get_text_embedding_llama(review_text, tokenizer, model, cache_dict, user_id)
        # 拼接评分向量与文本 embedding
        user_features = np.concatenate((rating_vector, text_embedding))
        features.append(user_features)
    # 保存缓存
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_dict, f)
    return np.array(features)

def get_text_embedding_llama(text, tokenizer, model, cache_dict=None, user_id=None):
    # 如果缓存存在则直接返回
    if cache_dict is not None and user_id is not None and user_id in cache_dict:
        return cache_dict[user_id]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).numpy().flatten()
    # 存到cache里
    if cache_dict is not None and user_id is not None:
        cache_dict[user_id] = embedding
    return embedding

class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        print("", "Initializing SimpleMLP with input dimension:", input_dim)
        super(SimpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 训练和测试神经网络的函数
def train_and_evaluate_nn(train_users, test_users, labels, simulated_data, tokenizer, model):
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print(f"Using device: {device}")

    X_train = extract_features_llama_with_cache(train_users, simulated_data, tokenizer, model)
    y_train = np.array([labels[u] for u in train_users])

    X_test = extract_features_llama_with_cache(test_users, simulated_data, tokenizer, model)
    y_test = np.array([labels[u] for u in test_users])

    model_nn = SimpleMLP(input_dim=X_train.shape[1]).to(device)  # 放到GPU

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_nn.parameters(), lr=0.001)

    batch_size = 128
    # 创建 TensorDataset 时先在 CPU，dataloader 会在迭代时加载到 GPU
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(f"Start Training")

    epochs = 100
    for epoch in range(epochs):
        model_nn.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)  # 放到GPU
            batch_y = batch_y.to(device)  # 放到GPU
            optimizer.zero_grad()
            outputs = model_nn(batch_X)
            l1_lambda = 0.001  # L1正则强度，你可以自己调
            l1_norm = sum(p.abs().sum() for p in model_nn.parameters())
            loss = criterion(outputs, batch_y) + l1_lambda * l1_norm
            # print("loss", loss.item())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # 测试
    model_nn.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = model_nn(X_test_tensor).cpu().numpy().flatten()
    predictions_binary = (predictions > 0.5).astype(int)

    accuracy = accuracy_score(y_test, predictions_binary)
    precision = precision_score(y_test, predictions_binary, zero_division=0)
    recall = recall_score(y_test, predictions_binary, zero_division=0)
    f1 = f1_score(y_test, predictions_binary, zero_division=0)
    auc = roc_auc_score(y_test, predictions)

    print("\nNN Test Set Metrics:")
    print(f"Accuracy:   {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 Score:  {f1:.2f}")
    print(f"AUC-ROC:   {auc:.2f}")

    return model_nn

# ------------------- 主程序 -------------------
if __name__ == "__main__":
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    client = OpenAI(api_key="Your Key", base_url="https://api.deepseek.com")

    item_info = load_item_info("item_info.txt")
    simulated_data = load_simulated_dataset("review_encoded.txt", item_info)
    train_users, train_labels = load_user_labels("train_half.txt")
    val_users, val_labels = load_user_labels("val_all.txt")
    test_users, test_labels = load_user_labels("test_all.txt")

    # 训练深度学习模型并保存参数
    mlp_model_path = "mlp_model.pth"
    # embedding缓存路径
    llama_path = "./LLaMA/"
    tokenizer = AutoTokenizer.from_pretrained(llama_path)
    model = AutoModel.from_pretrained(llama_path)
    tokenizer.pad_token = tokenizer.eos_token
    if os.path.exists(mlp_model_path):
        X_train = extract_features_llama_with_cache(train_users, simulated_data, tokenizer, model)
        model_nn = SimpleMLP(input_dim=X_train.shape[1]).to(device)
        model_nn.load_state_dict(torch.load(mlp_model_path))
    else:
        model_nn = train_and_evaluate_nn(
                        train_users,
                        test_users,
                        {**train_labels, **test_labels},
                        simulated_data,
                        tokenizer,
                        model
                    )
        torch.save(model_nn.state_dict(), mlp_model_path)

    PROMPTS = {
        "semantic_consistency": (
            "Please read all of the user's reviews below and summarize the overall consistency of viewpoints, emotions, or expression styles. For example: 'The user's comments are highly consistent in their tone and attitude, always displaying a rational and neutral stance.' Must in one English sentence."
        ),
        "sentiment_summary": (
            "Please read all of the user's reviews below and summarize the overall sentiment tendency and expressive style. For example: 'The comments are generally positive and the language is enthusiastic but not exaggerated.' Must in one English sentence."
        ),
        "opinion_diversity": (
            "Summarize the diversity of opinions in the user's reviews. For example: 'The user's opinions are highly diverse, expressing both positive and negative attitudes.' Must in one English sentence."
        ),
        "informativeness": (
            "Summarize how informative the user's reviews are. For example: 'The reviews are rich in useful details and provide specific product experiences.' Must in one English sentence."
        ),
        "detail_level": (
            "Summarize the level of detail in the user's reviews. For example: 'The reviews include many specific facts and vivid descriptions.' Must in one English sentence."
        ),
        "persuasive_strength": (
            "Summarize the overall persuasiveness of the user's reviews. For example: 'The user's arguments are clear and convincing.' Must in one English sentence."
        ),
        "emotional_intensity": (
            "Summarize the emotional intensity of the user's reviews. For example: 'The reviews are emotionally intense, often using strong adjectives.' Must in one English sentence."
        ),
        "subjectivity": (
            "Summarize the degree of subjectivity or objectivity in the user's reviews. For example: 'The comments are mostly subjective and based on personal feelings.' Must in one English sentence."
        ),
        "informativeness_bias": (
            "Summarize if the user's reviews have a bias toward certain types of information (e.g., technical details, personal stories, or general impressions). For example: 'The reviews focus mainly on technical features and product performance.' Must in one English sentence."
        ),
        "readability": (
            "Summarize the readability of the user's reviews. For example: 'The reviews are easy to read, well-structured, and free of spelling mistakes.' Must in one English sentence."
        ),
    }

    SEMANTIC_KEYS = [
        'semantic_consistency',
        'sentiment_summary',
        'opinion_diversity',
        'informativeness',
        'detail_level',
        'persuasive_strength',
        'emotional_intensity',
        'subjectivity',
        'informativeness_bias',
        'readability'
    ]
    
    # 提前计算所有metrics（包括MLP预测结果）
    metrics_path = "precomputed_metrics.json"
    required_metrics = [
        'avg_rating', 'std_rating', 'interaction_count', 'mlp_score', 'rating_variance',
        'max_rating', 'min_rating', 'median_rating', 'rating_range', 'positive_rating_ratio',
        'negative_rating_ratio', 'most_common_rating', 'average_review_length', 'std_review_length',
        'max_review_length', 'min_review_length', 'median_review_length', 'review_length_range'
    ] + SEMANTIC_KEYS

    # 加载已有metrics
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            precomputed_metrics = json.load(f)
    else:
        precomputed_metrics = {}

    def compute_user_metrics(user, simulated_data, tokenizer, model, model_nn, device):
        interactions = simulated_data[user]
        ratings = [i['rating'] for i in interactions]
        user_features = extract_features_llama_with_cache([user], simulated_data, tokenizer, model)[0]
        # 一定要保证输入和模型在同一device
        with torch.no_grad():
            user_features_tensor = torch.tensor(user_features, dtype=torch.float32, device=device)
            mlp_score = model_nn(user_features_tensor).item()
        all_reviews = " ".join([i['review_body'] for i in interactions if i.get('review_body', '')])
        review_lengths = [len(i['review_body'].split()) for i in interactions]
        metric_dict = {
            'avg_rating': float(np.mean(ratings)),
            'std_rating': float(np.std(ratings)),
            'interaction_count': len(interactions),
            'mlp_score': float(mlp_score),
            'rating_variance': float(np.var(ratings)),
            'max_rating': float(np.max(ratings)),
            'min_rating': float(np.min(ratings)),
            'median_rating': float(np.median(ratings)),
            'rating_range': float(np.ptp(ratings)),
            'positive_rating_ratio': float(np.mean(np.array(ratings) > 3)),
            'negative_rating_ratio': float(np.mean(np.array(ratings) < 3)),
            'most_common_rating': int(np.bincount(np.array(ratings).astype(int)).argmax()),
            'average_review_length': float(np.mean(review_lengths)),
            'std_review_length': float(np.std(review_lengths)),
            'max_review_length': float(np.max(review_lengths)),
            'min_review_length': float(np.min(review_lengths)),
            'median_review_length': float(np.median(review_lengths)),
            'review_length_range': float(np.ptp(review_lengths)),
            'semantic_consistency': None,
            'sentiment_summary': None,
            'opinion_diversity': None,
            'informativeness': None,
            'detail_level': None,
            'persuasive_strength': None,
            'emotional_intensity': None,
            'subjectivity': None,
            'informativeness_bias': None,
            'readability': None
        }
        return user, metric_dict, all_reviews

    def get_semantics(user, all_reviews, PROMPTS, client):
        # 并发获取所有10个语义指标
        from concurrent.futures import ThreadPoolExecutor
        results = {}

        def get_one(key):
            # 提供鲁棒的LLM调用（可按需加重试机制）
            return call_llm_until_response(PROMPTS[key], all_reviews, client)

        with ThreadPoolExecutor(max_workers=len(PROMPTS)) as executor:
            future2key = {executor.submit(get_one, key): key for key in SEMANTIC_KEYS}
            for future in as_completed(future2key):
                key = future2key[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    print(f"Semantic {key} failed for user {user}: {e}")
                    results[key] = ""
        return user, results

    metrics_path = "precomputed_metrics.json"
    BATCH_SIZE = 5
    all_users = train_users + val_users + test_users  # 你的变量
    total_user_num = len(all_users)

    # 加载已有metrics
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            precomputed_metrics = json.load(f)
    else:
        precomputed_metrics = {}

    print("Start precomputing metrics in batch mode...")

    for i in range(0, total_user_num, BATCH_SIZE):
        user_batch = all_users[i:i+BATCH_SIZE]
        batch_info = []
        # 先本地补齐所有可直接计算的指标
        for user in user_batch:
            need_semantics = any(
                (user not in precomputed_metrics) or
                (precomputed_metrics[user].get(key) in [None, ""])
                for key in SEMANTIC_KEYS
            )
            if need_semantics:
                user_metrics, metric_dict, all_reviews = compute_user_metrics(user, simulated_data, tokenizer, model, model_nn, device)
                batch_info.append((user, metric_dict, all_reviews))

        # 使用线程池并发调用 LLM（只处理需要的用户）
        if batch_info:
            with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
                futures = {
                    executor.submit(get_semantics, user, all_reviews, PROMPTS, client): user
                    for user, metric_dict, all_reviews in batch_info
                }
                results = {}
                for f in as_completed(futures):
                    user, sem_results = f.result()
                    results[user] = sem_results

            # 合并结果进 precomputed_metrics
            for user, metric_dict, all_reviews in batch_info:
                # 新增：单个user只补齐缺失部分
                if user in precomputed_metrics:
                    metric_dict = precomputed_metrics[user]
                sem_results = results[user]
                for key in SEMANTIC_KEYS:
                    if metric_dict.get(key) in [None, ""]:
                        metric_dict[key] = sem_results.get(key, "")
                precomputed_metrics[user] = metric_dict

            # 每批次处理后保存
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(precomputed_metrics, f, ensure_ascii=False, indent=2)
            print(f"Batch {i//BATCH_SIZE + 1} / {((total_user_num-1)//BATCH_SIZE)+1} finished & saved.")

    # 最后再存一次
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(precomputed_metrics, f, ensure_ascii=False, indent=2)

    print("All metrics precomputed. Proceeding to next steps...")

    # 所有metrics都给每层
    all_metrics = list(precomputed_metrics[train_users[0]].keys())

    # 生成经验数据库
    BATCH_SIZE = 10
    NUM_EPOCHS = 1
    buffer_path = "experience_text_buffer.jsonl"
    db_path = "experience_db.pkl"
    progress_path = "experience_progress.json"

    total_batches = (len(train_users) + BATCH_SIZE - 1) // BATCH_SIZE

    # 检查是否已全部完成
    status = load_experience_progress(NUM_EPOCHS, total_batches, progress_path)
    if status == "done" and os.path.exists(db_path):
        print("All epochs and batches completed, loading experience DB...")
    else:
        # 若没完成，续跑
        last_epoch, last_batch = status if isinstance(status, tuple) else (0, 0)
        global_agent = DecisionAgent(metrics=required_metrics, client=client)
        for epoch in range(last_epoch, NUM_EPOCHS):
            print(f"=== Training Epoch {epoch+1}/{NUM_EPOCHS} ===")
            user_batches = [train_users[i:i+BATCH_SIZE] for i in range(0, len(train_users), BATCH_SIZE)]
            start_batch = last_batch if epoch == last_epoch else 0
            for batch_idx, user_batch in enumerate(user_batches):
                if batch_idx < start_batch:
                    continue
                batch_metrics = [precomputed_metrics[u] for u in user_batch]
                experiences, recall = global_agent.batch_reasoning_and_summarize(
                    user_batch, batch_metrics, train_labels, tokenizer, model, client
                )
                save_experience_text(epoch, batch_idx, user_batch, batch_metrics, experiences, buffer_path)
                save_experience_progress(epoch, batch_idx, progress_path)
            last_batch = 0  # 下个 epoch 从0开始

        # 所有epoch后统一生成embedding
        print("All epochs completed, generating experience DB...")
        generate_experience_db_from_text(buffer_path, tokenizer, model, db_path)
        print("Experience DB built.")

    # print("All epochs completed, generating experience DB...")
    # generate_experience_db_from_text(buffer_path, tokenizer, model, db_path)
    # print("Experience DB built.")

    ############ 构建Agent层级结构#############
    # 假设 num_base = 16
    print("Start baseagents...")
    num_base = 16
    base_agents = [
        ImprovedBaseAgent(
            metrics=all_metrics,
            client=client,
            tokenizer=tokenizer,
            model=model,
            experience_db_path=db_path,
            user_list=train_users + val_users + test_users,
            user_metrics_dict=precomputed_metrics,
            precomputed_decision_path=f"base_agent_{i}_decisions.json",
            k=8,
            name=f"BaseAgent-{i}"   # 自动命名
        )
        for i in range(num_base)
    ]

    print(f"Initialized {len(base_agents)} BaseAgents.")

    # 组装 base_agents 已有
    global_base_agents = base_agents
    group_counts = [20, 16, 8]  # 每层组数自定义
    team_structure_records = []
    agents_hierarchy, leader_agents, team_structure_records = recursive_adaptive_structure_with_reuse(
        base_agents, train_users, train_labels, all_metrics, client,
        group_counts, min_group_size=2, max_group_size=8, max_layers=len(group_counts),
        tree_records=team_structure_records, global_base_agents=global_base_agents
    )

    # 1. 得到所有顶层LeaderAgent
    if isinstance(agents_hierarchy, list):
        top_leader_agents = agents_hierarchy
    else:
        top_leader_agents = [agents_hierarchy]

    print("顶层LeaderAgent数量:", len(top_leader_agents))  # 应该等于 group_counts[-1]

    # 结构树 team_structure_records 可以全程保存、可视化、或json序列化
    structure_tree_dicts = [export_team_structure(agent) for agent in top_leader_agents]
    with open('team_structure_tree.json', 'w', encoding='utf-8') as f:
        json.dump(structure_tree_dicts, f, ensure_ascii=False, indent=2)
        
    # 2. 决策Agent聚合所有顶层Leader
    decision_agent = DecisionAgent(
        metrics=all_metrics,
        client=client,
        subordinate_agents=top_leader_agents,  # 这里是所有顶层Leader
        base_agents=base_agents,
        tokenizer=tokenizer,
        model=model,
        k=5
    )

    print("DecisionAgent initialized with", len(decision_agent.subordinate_agents), "subordinate agents.")


    # ========== 2. 批量推理与refine主流程 ==========
    BATCH_SIZE = 10
    num_epoch = 3
    Refine_user_num = 64

    last_epoch, last_batch, user_status, config_dict = load_training_state(
        STATE_PATH, base_agents, leader_agents, decision_agent
    )

    for epoch in range(last_epoch, NUM_EPOCHS):
        user_batches = [train_users[i:i+BATCH_SIZE] for i in range(0, len(train_users), BATCH_SIZE)]
        for batch_idx, user_batch in enumerate(user_batches):
            if (epoch < last_epoch) or (epoch == last_epoch and batch_idx < last_batch):
                continue  # 跳过已完成部分

            batch_metrics = [precomputed_metrics[u] for u in user_batch]

            # 1. BaseAgent推理
            all_base_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(base_agents)) as executor:
                future_to_idx = {executor.submit(agent.batch_reasoning, user_batch, batch_metrics): idx
                                for idx, agent in enumerate(base_agents)}
                all_base_results = [None] * len(base_agents)
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    result, _, _ = future.result()
                    all_base_results[idx] = result
            # 对齐顺序
            all_base_results = [r for _, r in sorted(zip(range(len(base_agents)), all_base_results))]
            all_base_results_T = list(map(list, zip(*all_base_results)))  # [batch][n_base]

            all_leader_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(top_leader_agents)) as executor:
                futures = []
                for i, agent in enumerate(top_leader_agents):
                    batch_sub = all_base_results_T  # 这会下传到底层的 BaseAgent
                    futures.append(executor.submit(agent.batch_reasoning, user_batch, batch_metrics, batch_sub, 'Leader', i))
                for f in concurrent.futures.as_completed(futures):
                    result, _, _ = f.result()
                    all_leader_results.append(result)
            all_leader_results = [r for _, r in sorted(zip(range(len(top_leader_agents)), all_leader_results))]
            all_leader_results_T = list(map(list, zip(*all_leader_results)))  # [batch][n_top_leader]

            decision_results, _, _ = decision_agent.batch_reasoning(
                user_batch, batch_metrics, all_leader_results_T, agent_level='Decision', agent_idx=0
            )
            # 写入memory
            for idx, user_id in enumerate(user_batch):
                d = decision_results[idx]
                decision_agent.memory[user_id] = {
                    "decision": d.get("Decision"),
                    "reason": d.get("Reason"),
                    "metrics": batch_metrics[idx],
                    "is_correct": None,
                    "refine_experience": None
                }
                user_status.setdefault(user_id, {})["decision"] = d.get("Decision")

            print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Batch {batch_idx+1}/{len(user_batches)} processed.")

            if (batch_idx+1) % (Refine_user_num//BATCH_SIZE) == 0:
                decision_agent.experience.append(
                    decision_agent.summarize_epoch_experience()
                )
                # 只保留最新N条
                if len(decision_agent.experience) > 10:
                    decision_agent.experience = decision_agent.experience[-10:]
                save_decision_experience(decision_agent.experience, decision_agent.experience_path)

            print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Batch {batch_idx+1}/{len(user_batches)} processed. Experience updated.")

            gt_batch = [train_labels[u] for u in user_batch]
            d_pred_batch = [decision_agent.memory[u]["decision"] for u in user_batch]
            d_reason_batch = [decision_agent.memory[u]["reason"] for u in user_batch]
            feedbacks = [
                "correct" if p == g else "incorrect"
                for p, g in zip(d_pred_batch, gt_batch)
            ]
            # DecisionAgent refine（包含confidence和经验更新）
            decision_agent.batch_refine(
                user_batch,                # user_ids
                batch_metrics,             # user_metrics_list
                gt_batch,                  # ground_truths
                d_pred_batch,              # decisions
                d_reason_batch,            # reasons
                feedbacks,                 # feedbacks
                all_leader_results_T,      # batch_subordinate_results：传入 leaders 的 batch 结果
                agent_level="Decision",
                agent_idx=None,
                step="refine"
            )

            all_leader_agents = collect_all_leader_agents(top_leader_agents)

            leader_results_dict = {agent.agent_id: res for agent, res in zip(top_leader_agents, all_leader_results)}

            for agent in all_leader_agents:
                # 获取当前 agent 的 subordinate batch 结果
                batch_subordinate_results = get_subordinate_results_for_leader(agent, user_batch, all_base_results_T, leader_results_dict)
                # 采样 leader 决策和 reason
                if agent.agent_id in leader_results_dict:
                    agent_results = leader_results_dict[agent.agent_id]
                else:
                    agent_results = [{"Decision": 1, "Reason": "No cache."}] * len(user_batch)
                l_pred = [r["Decision"] for r in agent_results]
                l_reason = []
                for j, user_id in enumerate(user_batch):
                    leader_dec = agent_results[j].get("Decision", 1)
                    reason = sample_reason_recursive(agent, user_id, leader_dec)
                    l_reason.append(reason if reason is not None else "No supporting reason.")
                l_feedback = ["correct" if p == g else "incorrect" for p, g in zip(l_pred, gt_batch)]
                agent.batch_refine(
                    user_batch, batch_metrics, gt_batch, l_pred, l_reason, l_feedback, batch_subordinate_results, agent_level="Leader", val_users=val_users, val_labels=val_labels
                )

            print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Saving state after batch {batch_idx+1}/{len(user_batches)}...")
            save_training_state(
                STATE_PATH,
                epoch,
                batch_idx,
                base_agents,
                leader_agents,
                decision_agent,
                user_status=user_status
            )
            
            hard_downweight_nudging(
                top_leader_agents=top_leader_agents,
                all_leader_results=all_leader_results,
                gt_batch=gt_batch,
                lr=0.08  # 可调：0.02~0.10
            )
            print(f"[Saved state] epoch={epoch}, batch={batch_idx}")

        all_leader_agents = collect_all_leader_agents(top_leader_agents)
        structure_adaptive_search(
            top_leader_agents,        # 顶层Leader
            all_leader_agents,        # 所有层Leader
            base_agents,
            train_users, train_labels,
            val_users, val_labels
        )

        structure_tree_dicts = [export_team_structure(agent) for agent in top_leader_agents]
        with open('team_structure_tree.json', 'w', encoding='utf-8') as f:
            json.dump(structure_tree_dicts, f, ensure_ascii=False, indent=2)
        # structure_tree_dict = export_team_structure(top_leader_agents)
        # with open('team_structure_tree.json', 'w', encoding='utf-8') as f:
        #     json.dump(structure_tree_dict, f, ensure_ascii=False, indent=2)
        print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Batch {batch_idx+1}/{len(user_batches)} processed. structure_tree_dict updated.")

        print("==== Agent Hierarchy ====")
        for agent in top_leader_agents:
            print_team_structure(agent)

        # === 新增2：保存全局状态（训练断点、agent参数等）
        save_training_state(
            STATE_PATH,
            epoch,
            batch_idx,      # 注意 batch_idx 用最新的
            base_agents,
            leader_agents,
            decision_agent,
            user_status=user_status
        )

        for i, agent in enumerate(base_agents):
            acc, p, r, f, a = compute_agent_metrics(agent, train_users, train_labels)
            print(f"[BaseAgent-{i}] Accuracy: {acc:.2f}, Precision: {p:.2f}, Recall: {r:.2f}, F1: {f:.2f}, AUC: {a:.2f}")

        for i, agent in enumerate(leader_agents):
            acc, p, r, f, a = compute_agent_metrics(agent, train_users, train_labels)
            print(f"[{agent.agent_name}] Accuracy: {acc:.2f}, Precision: {p:.2f}, Recall: {r:.2f}, F1: {f:.2f}, AUC: {a:.2f}")

        acc, p, r, f, a = compute_agent_metrics(decision_agent, train_users, train_labels)
        print(f"[DecisionAgent] Accuracy: {acc:.2f}, Precision: {p:.2f}, Recall: {r:.2f}, F1: {f:.2f}, AUC: {a:.2f}")

    print("Training finished!")

    # ------------------ 测试阶段 ------------------
    print("=== Running Agent Testing ===")
    test_state_path = "test_process_state.json"

    if os.path.exists(test_state_path):
        with open(test_state_path, "r", encoding="utf-8") as f:
            test_state = json.load(f)
        finished_batches = set(test_state.get("finished_batches", []))
        batch_records = test_state.get("batch_records", {})
        print(f"[TEST] Recovering, {len(finished_batches)} batches already finished")
    else:
        finished_batches = set()
        batch_records = {}

    y_true, y_pred = [], []

    user_batches = [test_users[i:i+BATCH_SIZE] for i in range(0, len(test_users), BATCH_SIZE)]
    for batch_idx, user_batch in enumerate(user_batches):
        if str(batch_idx) in finished_batches:
            print(f"[TEST] Batch {batch_idx} already finished, skipping.")
            continue
        print(f"[TEST] Running batch {batch_idx} / {len(user_batches)}")
        batch_metrics = [precomputed_metrics[u] for u in user_batch]

        # 1. BaseAgent
        all_base_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(base_agents)) as executor:
            futures = [executor.submit(agent.batch_reasoning, user_batch, batch_metrics) for agent in base_agents]
            for f in concurrent.futures.as_completed(futures):
                result, _, _ = f.result()
                all_base_results.append(result)
        all_base_results = [r for _, r in sorted(zip(range(len(base_agents)), all_base_results))]
        all_base_results_T = list(map(list, zip(*all_base_results)))  # [batch][n_base]

        # 2. LeaderAgent
        all_leader_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(top_leader_agents)) as executor:
            futures = []
            for i, agent in enumerate(top_leader_agents):
                batch_sub = all_base_results_T
                futures.append(executor.submit(agent.batch_reasoning, user_batch, batch_metrics, batch_sub, 'Leader', i))
            for f in concurrent.futures.as_completed(futures):
                result, _, _ = f.result()
                all_leader_results.append(result)
        all_leader_results = [r for _, r in sorted(zip(range(len(top_leader_agents)), all_leader_results))]
        all_leader_results_T = list(map(list, zip(*all_leader_results)))

        # 3. DecisionAgent
        decision_results, _, _ = decision_agent.batch_reasoning(
            user_batch, batch_metrics, all_leader_results_T, agent_level='Decision', agent_idx=0
        )

        batch_y_true = [test_labels[u] for u in user_batch]
        batch_y_pred = [r.get("Decision", 1) for r in decision_results]
        y_true.extend(batch_y_true)
        y_pred.extend(batch_y_pred)

        batch_records[str(batch_idx)] = {
            "user_batch": user_batch,
            "metrics_batch": batch_metrics,
            "base_results": all_base_results,
            "leader_results": all_leader_results,
            "decision_results": decision_results,
            "y_true": batch_y_true,
            "y_pred": batch_y_pred,
        }
        finished_batches.add(str(batch_idx))

        # 保存进度
        with open(test_state_path, "w", encoding="utf-8") as f:
            json.dump({"finished_batches": list(finished_batches), "batch_records": batch_records}, f, ensure_ascii=False, indent=2)
        print(f"[TEST] Batch {batch_idx} finished and saved.")

    print("Test finished, aggregating results...")

    acc, precision, recall, f1, auc = metrics_from_records(batch_records)
    print(f"Test Accuracy:  {acc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1 Score:  {f1:.4f}")
    print(f"Test AUC-ROC:   {auc:.4f}")

    # Leaders
    num_leaders = len(top_leader_agents)  # 顶层leader数量
    for k in range(num_leaders):
        acc, p, r, f, a = leader_k_metrics_from_records(batch_records, k)
        print(f"[LeaderAgent-{k}] Acc: {acc:.2f}, P: {p:.2f}, R: {r:.2f}, F1: {f:.2f}, AUC: {a:.2f}")

    # Bases（可选，数量多请酌情打印）
    num_base = len(base_agents)
    for k in range(num_base):
        acc, p, r, f, a = base_k_metrics_from_records(batch_records, k)
        print(f"[BaseAgent-{k}]   Acc: {acc:.2f}, P: {p:.2f}, R: {r:.2f}, F1: {f:.2f}, AUC: {a:.2f}")
