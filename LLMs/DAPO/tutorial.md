### 关键要点
- 研究表明，DAPO（解耦剪切和动态采样策略优化）算法中的Token-level Policy Gradient Loss是一种用于大型语言模型（LLM）强化学习的损失函数，旨在处理长序列的训练。
- 证据倾向于认为，该损失函数通过为每个生成的token计算梯度，确保较长序列对梯度更具影响，从而提高训练稳定性。
- 它似乎可能涉及生成多个响应，使用旧策略计算奖励，并通过重要性比率和剪切机制优化策略。

---

### 什么是Token-level Policy Gradient Loss？
Token-level Policy Gradient Loss是DAPO算法的一部分，用于在强化学习中优化大型语言模型的策略。它的目标是最大化期望累积奖励，同时确保较长序列对梯度更新有更多影响，从而避免较短序列主导训练。

#### 如何工作？
该损失函数的工作流程包括：
- **生成响应**：为每个问题，使用旧策略（model_old）生成G个响应。
- **计算奖励**：为每个响应计算奖励R_i。
- **标准化优势**：通过减去奖励的均值并除以标准差，计算每个响应的优势值。
- **Token-level计算**：对每个响应中的每个token，计算新策略和旧策略下该token的概率比率（重要性比率），然后结合优势值和剪切机制计算损失。
- **平均损失**：将损失在token、响应和组之间平均化，最终返回负损失以便最小化（从而最大化目标函数）。

#### 为什么重要？
这种方法特别适用于长链推理（CoT）场景，帮助模型学习更复杂的推理模式，防止生成无意义或重复的文本，同时促进响应长度的健康增长。

---

### 意外的细节：奖励的标准化
一个可能出乎意料的细节是，优势值的计算是通过标准化整个组的奖励（减去均值除以标准差），而不是为每个token单独计算。这确保了训练的稳定性，但也意味着所有token共享相同的优势值，这在传统强化学习中可能不常见。

---

### 详细报告：DAPO中的Token-level Policy Gradient Loss分析

#### 背景与上下文
DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization）是一种开源的强化学习系统，旨在提升大型语言模型（LLM）的推理能力，尤其是在长链推理（CoT）场景下。其核心目标是通过强化学习从人类反馈中学习，优化模型以生成更有帮助、更准确的响应。2025年3月16日发表的论文“[DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/html/2503.14476v1)”详细描述了该算法，其中Token-level Policy Gradient Loss被认为是关键技术之一。

该损失函数的提出是为了解决长序列训练中的挑战，例如较长响应可能因token数量多而对梯度更新产生不成比例的影响，导致训练不稳定或生成无意义内容（如胡言乱语或重复词）。通过在token级别计算梯度，DAPO确保生成模式（无论响应长短）对奖励的影响能够被平等地促进或抑制。

#### 技术细节
根据论文提供的公式，Token-level Policy Gradient Loss的目标函数为：

\[
J_{DAPO}(\theta) = \mathbb{E}_{(q,a) \sim D, \{o_i\} \sim \pi_{\theta_{old}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left(r(\theta)_{i,t} \cdot \hat{A}_{i}, \text{clip}(r(\theta)_{i,t}, 1 - \epsilon_{\text{low}}, 1 + \epsilon_{\text{high}}) \cdot \hat{A}_{i}\right) \right]
\]

其中：
- \( q \) 是问题，\( a \) 是参考答案，\( D \) 是数据集。
- \( \{o_i\} \) 是使用旧策略 \( \pi_{\theta_{old}} \) 为每个问题生成的一组 \( G \) 个响应。
- \( r(\theta)_{i,t} = \frac{\pi_{\theta}(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,<t})} \) 是 token \( t \) 在响应 \( o_i \) 中的重要性比率，表示新策略相对于旧策略的概率比率。
- \( \hat{A}_{i} \) 是响应 \( i \) 的优势值，计算为 \( (R_i - \text{mean}(\{R_i\}_{i=1}^G)) / \text{std}(\{R_i\}_{i=1}^G) \)，其中 \( R_i \) 是响应 \( o_i \) 的奖励。
- \( \epsilon_{\text{low}} \) 和 \( \epsilon_{\text{high}} \) 是剪切范围参数，用于稳定训练，类似于 PPO 中的剪切机制。
- 公式中还有一个约束条件：\( 0 < |\{o_i | \text{is_equivalent}(a, o_i)\}| < G \)，可能表示生成的响应中与参考答案等价的响应数量应在 \( 0 \) 到 \( G \) 之间（不包括 \( 0 \) 和 \( G \)），但具体实现细节未明确。


#### 伪代码实现
以下是基于上述公式的伪代码实现，旨在捕捉Token-level Policy Gradient Loss的核心逻辑：

```python
def compute_dapo_loss(model, model_old, dataset_batch, G, epsilon_low, epsilon_high):
    total_loss = 0.0
    for group in dataset_batch:
        question, answer = group['question'], group['answer']
        # 使用旧策略生成G个响应
        responses = [generate_response(question, model_old) for _ in range(G)]
        # 计算每个响应的奖励
        rewards = [compute_reward(response) for response in responses]
        rewards = torch.stack(rewards)
        # 计算奖励的均值和标准差
        mean_R = torch.mean(rewards)
        std_R = torch.std(rewards)
        # 计算每个响应的优势值
        advantages = (rewards - mean_R) / (std_R + 1e-8)  # 避免除以零
        # 初始化该组的损失
        group_loss = 0.0
        for i in range(G):
            response = responses[i]
            advantage = advantages[i]
            # 分词响应
            response_tokens = tokenize(response)
            # 对每个token（不包括问题部分）
            for t in range(len(response_tokens)):
                # 状态是问题加上响应前t-1个token
                state = question + ' ' + ' '.join(response_tokens[:t])
                token = response_tokens[t]
                # 计算新策略下token的log概率
                input_ids = tokenize(state)
                output = model(input_ids)
                log_prob_new = output.logits[:, -1, token].squeeze()
                # 计算旧策略下token的log概率（不计算梯度）
                with torch.no_grad():
                    input_ids = tokenize(state)
                    output_old = model_old(input_ids)
                    log_prob_old = output_old.logits[:, -1, token].squeeze()
                # 计算重要性比率
                r_theta = torch.exp(log_prob_new - log_prob_old)
                # 计算剪切后的比率
                clipped_r = torch.clamp(r_theta, 1 - epsilon_low, 1 + epsilon_high)
                # 计算最小值（PPO风格的剪切目标）
                term = torch.min(r_theta * advantage, clipped_r * advantage)
                # 累加到组损失，平均化token数量
                group_loss += term / len(response_tokens)
        # 平均化G个响应
        group_loss /= G
        # 累加到总损失
        total_loss += group_loss
    # 平均化所有组
    total_loss /= len(dataset_batch)
    # 返回负损失以最小化（从而最大化J）
    return -total_loss
```

#### 实验结果与影响
根据论文，Token-level Policy Gradient Loss在AIME24基准测试中提升了模型性能，从41分提高到42分（表1）。实验还显示，该损失函数促进了生成熵和平均响应长度的增加（图4a和图4b，步骤0到8000）。这些结果表明，它对训练动态的稳定性至关重要，尤其是在长CoT场景下。

#### 比较与相关工作
与传统的PPO（Proximal Policy Optimization）相比，DAPO的Token-level Loss更注重token级别的梯度更新，而PPO通常在序列级别计算损失。论文“[DPO Meets PPO: Reinforced Token Optimization for RLHF](https://arxiv.org/html/2404.18922v1)”提出了一种类似的token-wise优化方法，强调从偏好数据中学习token级奖励函数，这与DAPO的理念有相似之处。

#### 实际应用与开源资源
DAPO的实现基于verl框架，训练代码和数据集在GitHub上开源（[DAPO GitHub](https://github.com/BytedTsinghua-SIA/DAPO)）。该系统在Qwen2.5-32B基模型上实现了AIME 2024的50分成绩，优于DeepSeek-R1-Zero-Qwen-32B的47分，且仅用了50%的训练步骤。

#### 表1：AIME24基准测试结果
| 模型/方法               | AIME24 avg@32 | 训练步骤比例 |
|-------------------------|---------------|--------------|
| DeepSeek-R1-Zero-Qwen-32B | 47            | 100%         |
| DAPO (Qwen2.5-32B)      | 50            | 50%          |

#### 结论
Token-level Policy Gradient Loss是DAPO算法的核心组件，通过在token级别优化策略，确保长序列训练的稳定性并促进模型性能提升。尽管其贡献相对较小（+2分AIME24），但在长CoT场景下至关重要。开源代码和数据集的可用性为研究社区提供了可复制的框架，推进了LLM强化学习的发展。

---

### 关键引用
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/html/2503.14476v1)
- [DPO Meets PPO: Reinforced Token Optimization for RLHF](https://arxiv.org/html/2404.18922v1)
