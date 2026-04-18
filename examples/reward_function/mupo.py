# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any

import numpy as np
from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

def diversity_reward(reward_inputs: list[dict[str, Any]]) -> list[float]:
    """
    For each example, compute the average distance between its embedding and all embeddings
    of *other groups* with the same uid, and normalize the reward to [0, 1] within each uid group.
    - "embedding": vector for this example
    - "uid": group id (shared among examples of the same problem)
    - "partition_id": to distinguish different groups for the same uid
    Returns: diversity_reward: list of float, one per example, normalized to [0, 1] (by uid)
    """

    uid_partition_to_indices = {}
    uid_to_indices = {}

    for idx, entry in enumerate(reward_inputs):
        uid = entry["uid"]
        partition_id = entry["partition_id"]
        key = (uid, partition_id)
        uid_partition_to_indices.setdefault(key, []).append(idx)
        uid_to_indices.setdefault(uid, []).append(idx)

    embeddings = [np.array(entry["embedding"]) for entry in reward_inputs]
    uids = [entry["uid"] for entry in reward_inputs]
    partition_ids = [entry["partition_id"] for entry in reward_inputs]

    raw_rewards = [0.0] * len(reward_inputs)
    # Populate raw_rewards: for each sample, the mean distance to other-partition group members with the same uid
    for i, (uid, partition_id, emb) in enumerate(zip(uids, partition_ids, embeddings)):
        # Indices for the same uid, but different partition_id
        other_indices = []
        for (cand_uid, cand_pid), indices in uid_partition_to_indices.items():
            if cand_uid == uid and cand_pid != partition_id:
                other_indices.extend(indices)
        if not other_indices:
            # No other group for this uid
            raw_rewards[i] = 0.0
            continue
        other_embs = [embeddings[j] for j in other_indices]
        emb_norm = np.linalg.norm(emb)
        if emb_norm == 0:
            avg_dist = 0.0
        else:
            one_minus_cos_sims = []
            for other_emb in other_embs:
                other_emb_norm = np.linalg.norm(other_emb)
                if other_emb_norm == 0:
                    one_minus_cos = 0.0
                else:
                    cos = float(np.dot(emb, other_emb) / (emb_norm * other_emb_norm))
                    one_minus_cos = 1.0 - cos
                one_minus_cos_sims.append(one_minus_cos)
            avg_dist = float(np.mean(one_minus_cos_sims)) if one_minus_cos_sims else 0.0
        raw_rewards[i] = avg_dist

    # Scale the raw_rewards by a multiplier (approximate to range [0, 1])
    normed = np.array(raw_rewards, dtype=np.float32) * 2
    return normed.tolist()



def compute_score(
    reward_inputs: list[dict[str, Any]], 
    format_weight: float = 0.1, 
    diversity_weight: float = 0.1
) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")
    # Only check the first example for uid and partition_id
    first = reward_inputs[0] if len(reward_inputs) > 0 else {}
    has_diversity = ("uid" in first and first["uid"] is not None and
                     "partition_id" in first and first["partition_id"] is not None)
    if has_diversity:
        diversity_rewards = diversity_reward(reward_inputs)
    else:
        diversity_rewards = [0.0] * len(reward_inputs)
        diversity_weight = 0.0  # do not include in overall score

    scores = []
    for idx, reward_input in enumerate(reward_inputs):
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        # Give diversity reward only if accuracy_reward == 1, else diversity is 0
        diversity_score = diversity_rewards[idx] if accuracy_score == 1 else 0.0
        overall_score = (
            (1 - format_weight - diversity_weight) * accuracy_score +
            format_weight * format_score +
            diversity_weight * diversity_score
        )
        scores.append(
            {
                "overall": overall_score,
                "format": format_score,
                "accuracy": accuracy_score,
                "diversity": diversity_rewards[idx],
            }
        )
    return scores
