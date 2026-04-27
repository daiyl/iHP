# Patient

- **Input**: Intervention Space, $A$.
- **Output**: State Space, $S$; Observation Space, $O$.
- **Probability transition model**: $p(s'|s, a)$, where $a \in A$, and $\{s, s'\} \in S$.
- **Decoder**: $dec(o|s)$, where $o \in O$.

## Models of patients: ##

1. **Delayed State Transition Model**. Assume the initial latent health state is the unhealthy state $s_0 = (0,0,1)$. When the same intervention $a = (0,0,1)$ is applied continuously for $n$ time steps, the latent health state transitions to the healthy state $(0,0,0)$. However, this represents only a partial state transition. If the intervention is discontinued at this point, the latent health state will revert back to $(0,0,1)$ after maintaining the intermediate state for $k$ time steps. Furthermore, if the intervention is sustained for $m$ time steps, where $m > n$, a complete state transition occurs and the latent health state remains stable at $(0,0,0)$. We define three mutually exclusive events based on the history of interventions up to time $t$:

	Event $E_{1a}$ (Complete Transition). $\exists i \in [0, t-m+1]$ such that $a_i = a_{i+1} = ... = a_{i+m-1} = (0, 0, 1)$.

	Event $E_{1b}$ (Partial Transition). $E_{1a}$ does not occur, and $\exists j \in [0, k-1]$ such that $a_{t-n-j+1} = ... = a_{t-j} = (0, 0, 1)$.
	
	Event $E_{1c}$ (No Effective Transition). None of the above events occur ($\lnot E_{1a}  \land \lnot E_{1b}$).
	
	The state transition probabilities can be represented as:
	
	$$p_1(s_{t+1}=(0,0,0)|E_{1a}) = 1, $$
	
	$$p_1(s_{t+1}=(0,0,0)|E_{1b}) = 1, $$
	
	$$p_1(s_{t+1}=(0,0,1)|E_{1c}) = 1. $$

2. **Periodic State Transition Model**. 


3. **Synergistic State Transition Model**.


4. **Tolerant State Transition Model**.


5. **Dependency-Induced Deterioration State Transition Model**.


6. **Allergy State Transition Model**.


7. **Antagonistic State Transition Model**.



### 📂 iHP/Patient/models/
- `base_model.py` - 统一接口基类
- `delayed_model.py` - 延迟状态转移模型
- `periodic_model.py` - 周期状态转移模型
- `synergistic_model.py` - 协同状态转移模型
- `tolerant_model.py` - 耐受状态转移模型
- `dependency_model.py` - 依赖恶化状态转移模型
- `allergy_model.py` - 过敏状态转移模型
- `antagonistic_model.py` - 拮抗状态转移模型

