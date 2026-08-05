# Waymo 평가지표 정리 (SMART 저장소 기준)

> 작성 목적: 이 저장소(`/home/airlab/SMART`)에서 모델을 평가할 때 **실제로 계산되는 지표**와,
> **Waymo 공식 리더보드가 계산하는 지표**를 구분해서 이해하기 위한 학습용 문서.
> 코드 수정 목적이 아니라 개념/수식 정리가 목적이다.

---

## 1. 개요

### 1.1 이 저장소가 관련된 두 트랙

SMART는 논문/README 기준으로 두 종류의 평가 맥락을 가진다.

| 트랙 | 무엇을 재는가 | 이 저장소 코드가 계산하는가 |
|---|---|---|
| **Waymo Open Sim Agents Challenge (WOSAC)** | 시뮬레이션된 다중 에이전트 미래가 "실제 주행 분포"와 얼마나 닮았는가 (realism) | **아니오** (아래 4장 참고) |
| **일반 motion prediction 지표** (minADE / minFDE 계열) | 예측 궤적이 GT 궤적과 얼마나 가까운가 (accuracy) | **부분적으로 예** (아래 2장, 단 공식 정의와 다름) |

README의 결과 표(SMART-tiny 0.7591 / SMART-large 0.7614 / SMART-zeroshot 0.7210)는
**WOSAC의 realism meta-metric 점수**이며, 이 값은 Waymo가 제공하는 별도 평가 라이브러리
(`waymo_open_dataset.wdl_limited.sim_agents_metrics`) 또는 Waymo 서버 채점으로 산출된 것이다.
**이 저장소의 `val.py`를 돌린다고 이 숫자가 나오지 않는다.**

### 1.2 데이터 시간 축 (WOMD 기준)

- 샘플링 주파수: **10 Hz**
- 과거(history): 11 스텝 = 1.1초 (`num_historical_steps: 11`)
- 미래(future): 80 스텝 = 8.0초 (`num_future_steps: 80`)
- SMART의 토큰 단위: `shift = 5` → **토큰 1개 = 0.5초(5스텝)**, 미래 8초 = **16개 토큰**
  - `smart/modules/agent_decoder.py:106` (`self.shift = 5`)
  - 토큰 vocabulary 크기 = `token_size: 2048` (`smart/tokens/cluster_frame_5_2048.pkl`)

### 1.3 관련 파일

```
/home/airlab/SMART/val.py                          # 검증 실행 스크립트 (README의 eval.py는 존재하지 않음)
/home/airlab/SMART/smart/model/smart.py            # LightningModule, validation_step / 로깅
/home/airlab/SMART/smart/metrics/__init__.py       # 내보내는 지표: AverageMeter, minADE, minFDE, TokenCls
/home/airlab/SMART/smart/metrics/min_ade.py        # minMultiADE(미사용), minADE(사용)
/home/airlab/SMART/smart/metrics/min_fde.py        # minMultiFDE(미사용), minFDE(사용)
/home/airlab/SMART/smart/metrics/next_token_cls.py # TokenCls
/home/airlab/SMART/smart/metrics/average_meter.py  # 단순 평균 누적기
/home/airlab/SMART/smart/metrics/utils.py          # topk / valid_filter / NMS 헬퍼
/home/airlab/SMART/smart/modules/agent_decoder.py  # forward(teacher forcing) / inference(rollout)
/home/airlab/SMART/configs/validation/validation_scalable.yaml
```

---

## 2. 현재 코드(`smart/metrics/`)에 구현된 지표

### 2.0 공통 구조

4개 지표 모두 `torchmetrics.Metric`을 상속하고, 동일한 누적 패턴을 쓴다.

```python
self.add_state('sum',   default=torch.tensor(0.0), dist_reduce_fx='sum')
self.add_state('count', default=torch.tensor(0),   dist_reduce_fx='sum')
...
def compute(self): return self.sum / self.count
```

- `dist_reduce_fx='sum'` 이므로 DDP 다중 GPU에서 `sum`/`count`가 각각 all-reduce된 뒤 나눠진다
  → **에폭 전체에 대한 마이크로 평균(micro average)**. 배치별 평균의 평균이 아니다.
- 즉 일반형은 다음과 같다.

$$
\text{Metric} \;=\; \frac{\sum_{\text{all samples}} \text{numerator}}{\sum_{\text{all samples}} \text{count}}
$$

`AverageMeter`는 이 패턴의 가장 단순한 형태로, `sum += val.sum()`, `count += val.numel()`
→ 임의 텐서의 원소 평균. (현재 SMART 모델에서는 인스턴스화되지 않아 실제로 쓰이지 않는다.)

---

### 2.1 minADE (`smart/metrics/min_ade.py`)

#### (a) 교과서적 정의

$K$개의 예측 궤적 $\hat{P}^{(k)} = \{\hat{p}^{(k)}_1,\dots,\hat{p}^{(k)}_T\}$과 GT $P=\{p_1,\dots,p_T\}$에 대해

$$
\mathrm{minADE}_K \;=\; \min_{k \in \{1,\dots,K\}} \; \frac{1}{T}\sum_{t=1}^{T}\bigl\lVert \hat{p}^{(k)}_t - p_t \bigr\rVert_2
$$

유효 마스크 $v_t \in \{0,1\}$를 쓰면

$$
\mathrm{minADE}_K \;=\; \min_{k} \; \frac{\sum_{t=1}^{T} v_t \lVert \hat{p}^{(k)}_t - p_t\rVert_2}{\sum_{t=1}^{T} v_t}
$$

#### (b) 이 저장소에 실제로 구현된 것

`min_ade.py`에는 클래스가 **두 개** 있다.

**(i) `minMultiADE` — 정의에 충실한 구현이지만 `__init__.py`에서 export되지 않고, 어디에서도 사용되지 않는다.**
`valid_filter` → `topk(max_guesses, pred, prob)` → `min_criterion`에 따라
`'ADE'`이면 K개 중 ADE 최소, `'FDE'`이면 **최종 스텝 오차가 가장 작은 모드 $k^*$를 먼저 고르고 그 모드의 ADE**를 취한다
(Argoverse/QCNet 계열의 관례).

**(ii) `minADE` — 실제로 사용되는 클래스.** `smart/model/smart.py:96`에서 `minADE(max_guesses=1)`로 생성된다.

핵심 코드(`min_ade.py:80-82`):

```python
eval_timestep = min(self.eval_timestep, pred.shape[1])          # self.eval_timestep = 70
self.sum += ((torch.norm(pred[:, :eval_timestep] - target[:, :eval_timestep], p=2, dim=-1)
              * valid_mask[:, :eval_timestep]).sum(dim=-1) / pred.shape[1]).sum()
self.count += valid_mask[:, :eval_timestep].any(dim=-1).sum()
```

`update()`의 상단(66-79행)에서 `valid_filter` / `topk` / min-over-K 로직은 **전부 주석 처리되어 있다.**
`max_guesses`, `prob`, `min_criterion`, `keep_invalid_final_step` 인자는 받기만 하고 **사용되지 않는다.**

따라서 실제 계산식은 (에이전트 $i$, 미래 스텝 $t$, $T_{\text{eval}}=70$, $T_{\text{pred}}=80$)

$$
\mathrm{minADE}_{\text{code}}
=\frac{\displaystyle\sum_{i \in \mathcal{A}} \frac{1}{T_{\text{pred}}}\sum_{t=1}^{T_{\text{eval}}} v_{i,t}\,\bigl\lVert \hat{p}_{i,t} - p_{i,t}\bigr\rVert_2}
{\bigl|\{\, i : \textstyle\sum_{t\le T_{\text{eval}}} v_{i,t} > 0 \,\}\bigr|}
$$

$$
T_{\text{eval}} = \min(70,\,T_{\text{pred}}) = 70,\qquad T_{\text{pred}} = 80
$$

#### (c) 정의와 어긋나는 지점 — 반드시 인지할 것

| 항목 | 교과서/공식 정의 | 이 코드 |
|---|---|---|
| **min over K** | $K=6$개 모드 중 최소 | **없음.** 단일 롤아웃 1개만 사용 (`max_guesses=1`, 게다가 인자 자체가 미사용). 사실상 `ADE`이지 `minADE`가 아니다 |
| **평가 구간** | 미래 전체 8.0 s (80 스텝) | **앞 7.0 s (70 스텝)만.** 마지막 1초(가장 어려운 구간)를 버린다 |
| **시간 평균 분모** | 유효 스텝 수 $\sum_t v_{i,t}$ (또는 최소한 $T_{\text{eval}}$) | **`pred.shape[1]` = 80.** 분자는 70스텝까지만 더하는데 분모는 80으로 나눈다 |
| **결과 편향** | — | 값이 체계적으로 **작게(좋게) 나온다.** 완전 유효한 에이전트 기준 대략 $70/80 = 0.875$ 배, 중간 결측이 있으면 더 작아짐 |
| **에이전트 집합** | WOMD `tracks_to_predict` (공식 지정 대상) | `data['agent']['valid_mask'][:, 10]`이 True인 **모든** 에이전트 (`smart.py:176`, `category == 3` 필터는 주석 처리됨) |

즉 **이 저장소가 찍는 `val_minADE`는 공식 minADE와 직접 비교 가능한 값이 아니다.**
"단일 샘플 롤아웃의 0~7 s 평균 변위오차를 80으로 나눠 스케일 다운한 값"으로 읽어야 한다.

#### (d) 어디서 로깅되는가

`smart/model/smart.py:192, 196`

```python
self.minADE.update(pred=pred_traj[eval_mask], target=gt[eval_mask], valid_mask=valid_mask[eval_mask])
self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
```

- `pred_traj`: `(N, 80, 2)` — `agent_decoder.inference()`가 16번의 재귀 스텝으로 5스텝씩 채운다 (`agent_decoder.py:459`)
- `gt`: `data['agent']['position'][:, 11:, :2]` — `(N, 80, 2)` (`agent_decoder.py:500`)
- `valid_mask`: `data['agent']['valid_mask'][:, 11:]` — `(N, 80)` (`smart.py:181`)
- `eval_mask`: `data['agent']['valid_mask'][:, 10]` — 현재 시점에 관측된 에이전트

> **중요:** 이 블록 전체가 `if self.inference_token:` 안에 있고 (`smart.py:177`),
> `self.inference_token`은 `smart.py:103`에서 **`False`로 하드코딩**되어 있으며 config로 바꿀 수 없다.
> **따라서 `python val.py`를 그냥 돌리면 `val_minADE` / `val_minFDE`는 아예 계산·로깅되지 않는다.**
> 로깅되는 것은 `val_cls_acc`와 `val_loss` 두 개뿐이다.

---

### 2.2 minFDE (`smart/metrics/min_fde.py`)

#### (a) 교과서적 정의

$$
\mathrm{minFDE}_K \;=\; \min_{k \in \{1,\dots,K\}} \bigl\lVert \hat{p}^{(k)}_{T} - p_{T} \bigr\rVert_2
$$

여기서 $T$는 **예측 지평의 마지막 유효 스텝**(WOMD marginal 기준 $t=8.0$ s).

#### (b) 이 저장소에 실제로 구현된 것

`minMultiFDE`(미사용, export 안 됨)는 정의에 충실하다:
`valid_filter` → `topk` → 각 에이전트의 마지막 유효 스텝 인덱스
`inds_last = argmax_t (v_t \cdot t)` 를 구해 그 시점 오차의 K개 중 최소값을 취한다.

실제 사용되는 `minFDE` (`smart/model/smart.py:97`에서 `minFDE(max_guesses=1)`), 핵심 코드(`min_fde.py:55-58`):

```python
eval_timestep = min(self.eval_timestep, pred.shape[1]) - 1       # = min(70, 80) - 1 = 69
self.sum += ((torch.norm(pred[:, eval_timestep-1:eval_timestep] - target[:, eval_timestep-1:eval_timestep],
                         p=2, dim=-1)
              * valid_mask[:, eval_timestep-1].unsqueeze(1)).sum(dim=-1)).sum()
self.count += valid_mask[:, eval_timestep-1].sum()
```

`eval_timestep - 1 = 68` 이므로 슬라이스는 `pred[:, 68:69]`, 마스크는 `valid_mask[:, 68]`.
미래 인덱스 68 → 시각 $(68+1)\times 0.1 = \mathbf{6.9\ s}$.

$$
\mathrm{minFDE}_{\text{code}}
=\frac{\displaystyle\sum_{i\in\mathcal{A}} v_{i,68}\,\bigl\lVert \hat{p}_{i,68} - p_{i,68}\bigr\rVert_2}
{\displaystyle\sum_{i\in\mathcal{A}} v_{i,68}}
$$

#### (c) 정의와 어긋나는 지점

| 항목 | 교과서/공식 정의 | 이 코드 |
|---|---|---|
| **min over K** | $K=6$ 중 최소 | **없음.** 단일 롤아웃. `max_guesses`/`prob` 미사용 |
| **"Final" 시점** | $t = 8.0$ s (인덱스 79) | **$t = 6.9$ s (인덱스 68).** `-1`이 두 번 적용되어 70도 69도 아닌 68이 된다 |
| **에이전트별 마지막 유효 스텝 처리** | 각 에이전트의 실제 마지막 유효 스텝 사용 | 전 에이전트 공통 고정 인덱스 68. 68에서 무효한 에이전트는 **집계에서 통째로 제외** |
| **결과 편향** | — | 8.0 s 대신 6.9 s를 재므로 값이 **작게(좋게)** 나온다 |

이름은 FDE(Final Displacement Error)지만 **최종 시점이 아니다.** 실질적으로는 "$t=6.9$ s 변위오차"다.

#### (d) 어디서 로깅되는가

`smart/model/smart.py:193, 197` — minADE와 동일하게 `if self.inference_token:` (기본값 `False`) 안에 있다.

---

### 2.3 Next-token Classification Accuracy (`smart/metrics/next_token_cls.py`)

#### (a) 정의

SMART는 궤적 회귀가 아니라 **이산 토큰의 다음 토큰 예측(next-token prediction)** 문제로 모델링한다.
따라서 가장 직접적인 지표는 **다음 모션 토큰을 얼마나 맞추는가**이다.

top-$k$ 정확도:

$$
\mathrm{Acc}@k
=\frac{\displaystyle\sum_{(i,t)\in\mathcal{V}} \mathbb{1}\!\left[\, y_{i,t} \in \mathrm{TopK}_k\bigl(\hat{y}_{i,t}\bigr) \right]}
{\bigl|\mathcal{V}\bigr|}
$$

- $y_{i,t} \in \{0,\dots,2047\}$: 에이전트 $i$의 시각 $t$에서의 GT 다음 토큰 인덱스
- $\mathcal{V}$: 평가 대상 (에이전트, 토큰스텝) 쌍의 집합
- 토큰 스텝은 0.5초 단위이므로 시나리오 1개당 에이전트별 최대 약 18개 스텝(과거 2 + 미래 16)

#### (b) 코드 구현

```python
class TokenCls(Metric):
    def update(self, pred, target, valid_mask=None):
        target = target[..., None]
        acc = (pred[:, :self.max_guesses] == target).any(dim=1) * valid_mask
        self.sum += acc.sum()
        self.count += valid_mask.sum()
```

- `smart/model/smart.py:98`에서 `TokenCls(max_guesses=1)` → **top-1 정확도**
- `pred`는 `agent_decoder.py:337`의 `torch.topk(softmax(logits), k=10)` 결과라서 `(M, 10)` 형태.
  `max_guesses=1`이므로 `pred[:, :1]`, 즉 **argmax만** 쓴다.
  (`max_guesses`를 5로 바꾸면 top-5 정확도가 되고, 10까지는 코드 수정 없이 가능하다.)
- `validation_step`에서 이미 마스크로 걸러진 텐서를 넘기므로
  (`smart.py:171-172`: `pred=next_token_idx[next_token_eval_mask]`, `valid_mask=next_token_eval_mask[next_token_eval_mask]`)
  전달되는 `valid_mask`는 **전부 True**이고, 실질적으로 분모는 `M` = 유효 쌍의 개수다. 마스크는 중복 적용이라 무의미하다.

#### (c) `next_token_eval_mask`의 정의 (`agent_decoder.py:340-342`)

```python
next_token_eval_mask = mask.clone()
next_token_eval_mask = next_token_eval_mask * next_token_eval_mask.roll(shifts=-1, dims=1) \
                                            * next_token_eval_mask.roll(shifts=1, dims=1)
next_token_eval_mask[:, -1] = False
```

$$
\mathcal{V} = \{(i,t) \;:\; m_{i,t-1} \wedge m_{i,t} \wedge m_{i,t+1},\;\; t \neq T_{\text{last}}\}
$$

즉 **직전·현재·직후 토큰이 모두 유효한 스텝만** 평가한다. 마지막 토큰 스텝은 항상 제외
(다음 토큰이 존재하지 않으므로).

#### (d) 로깅 & 중요한 성질

`smart/model/smart.py:171-173`

```python
self.TokenCls.update(pred=next_token_idx[next_token_eval_mask],
                     target=next_token_idx_gt[next_token_eval_mask],
                     valid_mask=next_token_eval_mask[next_token_eval_mask])
self.log('val_cls_acc', self.TokenCls, prog_bar=True, on_step=False, on_epoch=True,
         batch_size=1, sync_dist=True)
```

> **이것이 `val.py` 기본 설정에서 실제로 나오는 유일한 정확도 지표다.**

핵심 성질 두 가지:

1. **Teacher forcing 상태에서 측정된다.** `validation_step`은 `self(data)` = `encoder.forward()`를 호출하고,
   이 경로는 GT 토큰 시퀀스를 입력으로 받는다 (`agent_decoder.forward`). 자기회귀 롤아웃(`inference()`)이 아니다.
   따라서 **누적 오차(compounding error)를 전혀 반영하지 않는다.** 실제 시뮬레이션 품질의 상한 지표에 가깝다.
2. **2048-way 분류**의 top-1 정확도다. 무작위 추측 기준선은 $1/2048 \approx 0.049\%$.
   또한 토큰 사전은 클러스터링으로 만들어졌기 때문에 인접 토큰끼리 기하학적으로 매우 유사하다
   → **top-1 오답이 곧 큰 궤적 오차를 의미하지 않는다.** 정확도 절대값보다 상대 비교용으로 보는 편이 안전하다.

#### (e) 함께 로깅되는 손실

`val_loss` = `val_cls_loss` = label smoothing 0.1이 적용된 cross-entropy (`smart.py:101, 169`)

$$
\mathcal{L} = -\frac{1}{|\mathcal{V}|}\sum_{(i,t)\in\mathcal{V}}\sum_{c=1}^{2048} \tilde{y}_c \log \mathrm{softmax}\bigl(\hat{y}_{i,t}\bigr)_c,
\qquad \tilde{y}_c = (1-\epsilon)\,\mathbb{1}[c = y_{i,t}] + \frac{\epsilon}{2048},\;\; \epsilon = 0.1
$$

label smoothing 때문에 **완벽한 모델이라도 `val_loss`가 0으로 수렴하지 않는다.** 하한이 존재한다.

---

### 2.4 요약: `val.py` 기본 실행 시 나오는 것

```bash
python val.py --config configs/validation/validation_scalable.yaml --pretrain_ckpt <ckpt>
```

| 로그 키 | 계산되는가 | 내용 |
|---|---|---|
| `val_cls_acc` | ✅ | 2048-way 다음 토큰 top-1 정확도 (teacher forcing) |
| `val_loss` | ✅ | label-smoothed CE loss |
| `val_minADE` | ❌ | `self.inference_token = False` (`smart.py:103`)이라 블록 진입 안 함 |
| `val_minFDE` | ❌ | 동일 |
| WOSAC realism | ❌ | 이 저장소에 계산 코드 없음 |

`val_minADE`/`val_minFDE`를 보려면 `smart/model/smart.py:103`의 `self.inference_token`을 `True`로
바꿔야 하며, 그렇게 하더라도 위 2.1(c)/2.2(c)의 정의 차이는 그대로 남는다.

#### 그 외 평가 재현성 관련 주의

`smart/model/smart.py:70`의 `self.noise = True`는 `match_token_map()`에서
가장 가까운 map token 대신 **top-8 후보 중 하나를 무작위 샘플링**하게 만든다 (`smart.py:278-281`).
이 코드는 `training_step`뿐 아니라 `validation_step`에서도 실행되므로,
**검증 지표에 확률적 변동이 섞인다.** 또 `inference()`의 토큰 샘플링도
`torch.multinomial(topk_prob, 1)` (beam_size=5, `agent_decoder.py:437, 454`)로 확률적이다.
정확한 재현이 필요하면 이 두 지점을 인지하고 있어야 한다.

---

## 3. Waymo 공식 Motion Prediction Challenge 지표

이 저장소는 이 지표들을 **계산하지 않는다.** 공식 계산은
`waymo_open_dataset.metrics.ops.py_metrics_ops.motion_metrics` (C++ op)가 담당한다.

### 3.1 공식 설정 (`MotionMetricsConfig`)

Waymo 튜토리얼(`tutorial/tutorial_motion.ipynb`)에 실린 공식 챌린지 config:

```
track_steps_per_second: 10          # GT는 10 Hz
prediction_steps_per_second: 2      # 제출 예측은 2 Hz (0.5초 간격, 총 16점)
track_history_samples: 10
track_future_samples: 80
max_predictions: 6                  # K = 6
speed_lower_bound: 1.4              # m/s
speed_upper_bound: 11.0             # m/s
speed_scale_lower: 0.5
speed_scale_upper: 1.0
step_configurations { measurement_step: 5,  lateral_miss_threshold: 1.0, longitudinal_miss_threshold: 2.0 }  # t = 3 s
step_configurations { measurement_step: 9,  lateral_miss_threshold: 1.8, longitudinal_miss_threshold: 3.6 }  # t = 5 s
step_configurations { measurement_step: 15, lateral_miss_threshold: 3.0, longitudinal_miss_threshold: 6.0 }  # t = 8 s
```

시각 환산: $t = (\texttt{measurement\_step} + 1)/\texttt{prediction\_steps\_per\_second}$
→ step 5 → 3 s, step 9 → 5 s, step 15 → 8 s.

모든 지표는 **{3 s, 5 s, 8 s} × {VEHICLE, PEDESTRIAN, CYCLIST}** 조합별로 계산된 뒤 평균된다.

### 3.2 minADE

$$
\mathrm{minADE}(T) \;=\; \underset{i}{\mathbb{E}}\left[\;\min_{k\le 6}\; \frac{1}{|\mathcal{T}_i(T)|}\sum_{t\in\mathcal{T}_i(T)} \bigl\lVert \hat{p}^{(k)}_{i,t} - p_{i,t}\bigr\rVert_2 \right]
$$

`motion_metrics.proto`의 서술: *"the average difference from the ground truth in meters is computed
up to the measurement time step ... for all trajectory predictions for that object.
The value with the minimum error is kept (minADE)."*

핵심: **측정 시점 $T$까지 누적**하며, $\mathcal{T}_i(T)$는 유효 스텝만 포함한다.

### 3.3 minFDE

$$
\mathrm{minFDE}(T) \;=\; \underset{i}{\mathbb{E}}\left[\;\min_{k\le 6}\; \bigl\lVert \hat{p}^{(k)}_{i,T} - p_{i,T}\bigr\rVert_2 \right]
$$

**측정 시점 $T$ 정확히 그 스텝**의 오차. $T \in \{3, 5, 8\}$ s.

### 3.4 Miss Rate (MR)

공식 MR은 단순 유클리드 임계값이 아니라 **GT 진행 방향 기준 종/횡 분해 + 초기 속도 스케일링**을 쓴다.

에이전트 $i$의 시각 $T$에서, GT 헤딩 방향을 기준으로 오차를 종방향(longitudinal) $e^{\text{lon}}$,
횡방향(lateral) $e^{\text{lat}}$로 분해한다. 초기 속도 $s_i$에 대한 스케일 계수는

$$
\alpha(s_i) =
\begin{cases}
0.5 & s_i \le 1.4\ \mathrm{m/s} \\[4pt]
0.5 + 0.5\cdot\dfrac{s_i - 1.4}{11.0 - 1.4} & 1.4 < s_i < 11.0 \\[8pt]
1.0 & s_i \ge 11.0\ \mathrm{m/s}
\end{cases}
$$

(= `speed_scale_lower`=0.5 와 `speed_scale_upper`=1.0 사이 선형 보간)

$k$번째 예측이 "hit"이려면

$$
\bigl|e^{\text{lat},(k)}_{i,T}\bigr| \le \alpha(s_i)\cdot \tau^{\text{lat}}(T)
\quad\wedge\quad
\bigl|e^{\text{lon},(k)}_{i,T}\bigr| \le \alpha(s_i)\cdot \tau^{\text{lon}}(T)
$$

| $T$ | $\tau^{\text{lat}}$ | $\tau^{\text{lon}}$ |
|---|---|---|
| 3 s | 1.0 m | 2.0 m |
| 5 s | 1.8 m | 3.6 m |
| 8 s | 3.0 m | 6.0 m |

$$
\mathrm{MR}(T) = \frac{\bigl|\{\,i : \text{6개 예측 중 hit이 하나도 없음}\,\}\bigr|}{\bigl|\{\text{평가 대상 에이전트}\}\bigr|}
$$

즉 **낮을수록 좋다.**

### 3.5 mAP / soft mAP

`motion_metrics.proto`의 서술을 그대로 옮기면:

- GT 궤적을 **행동 버킷(trajectory shape bucket)** 으로 분류한다
  (straight, straight-left, straight-right, left, right, left u-turn, right u-turn, stationary 등).
  2024 챌린지에서 이 버킷 분류 로직이 개선되었다고 Waymo가 공지했다.
- 각 예측을 confidence 내림차순으로 정렬한 뒤, 3.4의 hit 판정으로 TP/FP를 매긴다.
  **에이전트당 TP는 최대 1개**이고, 나머지 hit들은 **FP로 계산된다(mAP)**.
- 버킷별 Average Precision을 PASCAL VOC(2010 이후, all-points interpolation) 방식으로 구하고,
  버킷 전체에 대해 평균 → **mAP**.

$$
\mathrm{AP}_b = \int_0^1 p_b(r)\,dr \quad(\text{all-points interpolation}),
\qquad
\mathrm{mAP} = \frac{1}{|B|}\sum_{b\in B}\mathrm{AP}_b
$$

- **soft mAP**: *"duplicate true positives per ground truth trajectory are ignored rather than
  counted as false positives."* 즉 같은 GT에 대한 **중복 hit을 FP로 처벌하지 않고 무시**한다.
  → 다중 모드를 많이 내는 모델에 덜 가혹하다.

**WOMD Motion Prediction 리더보드의 1차 랭킹 지표는 soft mAP, 2차는 Miss Rate이다.**
minADE/minFDE는 보조 지표다.

### 3.6 Overlap Rate

*"Overlaps are detected as any intersection of the bounding boxes of the highest confidence
predicted object trajectory with those of any other valid object at the same time step ...
If one or more overlaps occur up to the measurement step it is considered a single overlap measurement."*

$$
\mathrm{OverlapRate}(T) = \frac{\bigl|\{\,i:\ \exists\, t \le T,\ \exists\, j\neq i,\ \mathrm{BBox}(\hat{p}^{(1)}_{i,t}) \cap \mathrm{BBox}(p_{j,t}) \neq \emptyset \,\}\bigr|}{|\{\text{평가 대상 에이전트}\}|}
$$

- **최고 confidence 궤적(top-1)만** 사용한다는 점이 중요하다.
- 상대는 GT의 다른 valid 객체다.

### 3.7 이 저장소 구현과의 차이 (요약표)

| 항목 | Waymo 공식 | 이 저장소 `smart/metrics/` |
|---|---|---|
| K (모드 수) | **6** | **1** (min-over-K 로직 없음) |
| 예측 샘플링 | 2 Hz (16점) | 10 Hz (80점) |
| 측정 시점 | 3 s / 5 s / 8 s 각각 | ADE: 0–7 s 누적 / FDE: 6.9 s 단일 시점 |
| ADE 시간 분모 | 유효 스텝 수 | **80 (고정, 분자 구간과 불일치)** |
| 평가 대상 | 공식 `tracks_to_predict` | `WaymoTargetBuilder.score_trained_vehicle`로 **자체 선정**한 ego 100 m 이내 최대 32대 (무작위 서브샘플 포함, `smart/transforms/target_builder.py:93-115`) — 게다가 `validation_step`은 `category == 3` 필터마저 주석 처리해 valid한 모든 에이전트를 씀 |
| 타입별 분리 | VEHICLE/PED/CYCLIST별 산출 | 없음 (전부 합산) |
| MR / mAP / soft mAP / Overlap | 계산함 | **계산 안 함** |
| 랭킹 지표 | soft mAP | — |

> **결론: 이 저장소의 `val_minADE`, `val_minFDE` 숫자를 논문/리더보드의 minADE, minFDE와
> 직접 비교하면 안 된다.** 내부 학습 진행도 추적용으로만 사용하고,
> 공식 수치가 필요하면 `waymo_open_dataset` 평가 라이브러리로 별도 계산해야 한다.

---

## 4. Waymo Open Sim Agents Challenge (WOSAC) 지표

README의 0.7591 / 0.7614 / 0.7210이 바로 이 지표(realism meta-metric)다.

### 4.1 과제 설정 (2024 기준)

| 항목 | 값 |
|---|---|
| 과거 컨텍스트 | 1.1 s (11 스텝 @ 10 Hz) |
| 시뮬레이션 지평 | **8.0 s (80 스텝 @ 10 Hz)** |
| 롤아웃 수 | **K = 32** joint futures / 시나리오 |
| 시뮬레이션 대상 | 씬의 **모든** valid 에이전트 (marginal prediction과 달리 일부가 아님) |
| 출력 | 각 에이전트의 $(x, y, z, \text{heading})$ 시퀀스 |
| 추론 주기 | 최소 1 Hz 허용, 출력은 10 Hz로 보간 제출 가능 |

"joint"라는 점이 핵심이다 — 롤아웃 하나는 **씬 전체 에이전트의 일관된 하나의 미래**여야 한다.

### 4.2 기본 아이디어: 분포 매칭 (NLL)

WOSAC는 "GT에 얼마나 가까운가"가 아니라 **"시뮬레이터가 만든 분포가 실제 로그 분포와 얼마나 닮았는가"** 를 잰다.

제출은 샘플 32개일 뿐 해석적 확률밀도가 아니므로, Waymo는 각 (에이전트 $i$, 시각 $t$, feature $j$)마다
**32개 샘플로 히스토그램(또는 KDE)을 적합**하고, 그 분포 하에서 **로그 GT 값의 likelihood**를 평가한다.
이산 히스토그램에는 **additive smoothing (pseudocount = 0.1)** 을 적용해 0 확률을 방지한다.

시각별 NLL을 유효 마스크로 평균한 뒤 지수화해 [0,1] 범위 점수로 만든다:

$$
m_{i,j} \;=\; \exp\!\left(-\frac{1}{\sum_t \mathbb{1}[v_{i,t}]}\sum_t \mathbb{1}[v_{i,t}]\cdot \mathrm{NLL}_{i,j,t}\right)
$$

$\mathrm{NLL}_{i,j,t} = -\log \hat{q}_{i,j,t}\bigl(\text{GT feature 값}\bigr)$, $\hat{q}$는 32개 샘플로 추정한 분포.

### 4.3 Realism Meta-metric

$$
\mathcal{M}^{K} \;=\; \frac{1}{N\,}\sum_{i=1}^{N} \sum_{j=1}^{M} w_j \, m^{K}_{i,j},
\qquad \sum_j w_j = 1
$$

$N$ = 평가 대상 에이전트 수, $M$ = component metric 수, $w_j$ = 가중치.
**높을수록 좋다 (최대 1.0).**

### 4.4 Component metric과 실제 가중치 (2024 공식 config)

아래는 `waymo-open-dataset` 저장소의
`src/waymo_open_dataset/wdl_limited/sim_agents_metrics/challenge_2024_config.textproto`
**원문 값**이다 (추측 아님).

| 그룹 | Feature | 추정 분포 | 히스토그램 범위 | bins | **weight** |
|---|---|---|---|---|---|
| Kinematic | `linear_speed` | histogram | [0.0, 25.0] | 10 | **0.05** |
| Kinematic | `linear_acceleration` | histogram | [-12.0, 12.0] | 11 | **0.05** |
| Kinematic | `angular_speed` | histogram | [-0.628, 0.628] | 11 | **0.05** |
| Kinematic | `angular_acceleration` | histogram | [-3.14, 3.14] | 11 | **0.05** |
| Interactive | `distance_to_nearest_object` | histogram | [-5.0, 40.0] | 10 | **0.10** |
| Interactive | `collision_indication` | **Bernoulli** | — | — | **0.25** |
| Interactive | `time_to_collision` | histogram | [0.0, 5.0] | 10 | **0.10** |
| Map-based | `distance_to_road_edge` | histogram | [-20.0, 40.0] | 10 | **0.10** |
| Map-based | `offroad_indication` | **Bernoulli** | — | — | **0.25** |
| (2025 추가) | `traffic_light_violation` | Bernoulli | — | — | **0.00** (2024에선 미사용) |

가중치 합 = $0.05\times4 + 0.10 + 0.25 + 0.10 + 0.10 + 0.25 + 0.00 = 1.00$.

**해석상 가장 중요한 사실: 충돌(0.25) + 오프로드(0.25) = 전체의 50%.**
운동학 4개를 다 합쳐도 0.20에 불과하다. WOSAC 점수는 사실상 **"충돌하지 않고 도로 위에 머무는가"** 가 지배한다.

모든 histogram 지표는 `independent_timesteps: true`, `additive_smoothing_pseudocount: 0.1`.
`independent_timesteps: true`는 각 타임스텝을 독립적으로 평가한 뒤 평균한다는 뜻이다.

각 feature의 의미:

- **linear speed**: 스텝 간 $(x,y,z)$ 변화량의 크기
- **angular speed**: 스텝 간 yaw의 부호 있는 변화량
- **linear acceleration**: speed 크기의 부호 있는 변화량
- **angular acceleration**: angular speed의 부호 있는 변화량
- **distance to nearest object**: 다른 에이전트까지의 최소 거리 (박스 간 거리이므로 음수 가능 → 범위 하한 −5.0)
- **collision indication**: 다른 객체와 충돌했는지 여부 (boolean)
- **time to collision**: 현재 상태 유지 시 충돌까지 남은 시간(초)
- **distance to road edge**: 주행 가능 영역 경계까지의 최소 거리 (도로 밖이면 음수 → 범위 하한 −20.0)
- **offroad indication**: 주행 가능 영역을 벗어났는지 여부 (boolean)

### 4.5 함께 보고되는 보조 지표

WOSAC 리더보드는 realism meta-metric 외에
- 그룹별 점수 (**kinematic / interactive / map-based** likelihood)
- **minADE**: 32개 롤아웃 중 GT에 가장 가까운 것의 ADE
  → 다양성만 크고 정확도가 없는 모델을 걸러내는 sanity check 역할

를 함께 표시한다. 여기서의 minADE는 $K=32$, 8초 전체 기준이므로
**2장의 코드 minADE(K=1, 0–7 s, 분모 80)와 또 다르다.**

### 4.6 이 저장소는 WOSAC 지표를 계산하지 않는다

확인된 사실:

- `smart/model/smart.py:17`에서 `from waymo_open_dataset.protos import sim_agents_submission_pb2`를 import하고,
  `joint_scene_from_states()` (`smart.py:41-50`)라는 제출 포맷 변환 헬퍼가 정의되어 있다.
- 그러나 `on_validation_start`에서 초기화되는 `self.scenario_rollouts` (`smart.py:202`)는
  **어디에서도 채워지지 않으며**, 제출 파일 작성 코드도, `sim_agents_metrics` 호출도 저장소에 없다.
- `self.rollout_num = 1` (`smart.py:104`)로 설정되어 있고 실제로 사용되지도 않는다.
  WOSAC가 요구하는 K=32 롤아웃 루프가 구현되어 있지 않다.
- 참고로 `waymo_open_dataset` 패키지는 현재 이 환경에 설치되어 있지도 않다
  (`import waymo_open_dataset` → `ModuleNotFoundError`).
  즉 `smart/model/smart.py`를 import하려면 먼저 Waymo API를 설치해야 한다.

**따라서 README의 0.7591 등은 이 코드의 출력이 아니라, 별도의 제출 파이프라인 + Waymo 공식 평가로 산출된 값이다.**

직접 재현하려면 대략 다음이 필요하다:

1. `waymo-open-dataset-tf-*` 설치
2. `encoder.inference()`를 시나리오당 32회 반복 (현재 beam sampling이 확률적이므로 매번 다른 롤아웃이 나옴)
3. `pos_a`/`head_a` + $z$ 좌표를 `joint_scene_from_states()`로 `JointScene` 변환
4. `ScenarioRollouts` → `SimAgentsChallengeSubmission` 직렬화
5. `waymo_open_dataset.wdl_limited.sim_agents_metrics.metrics.compute_scenario_metrics_for_bundle()` 호출
   (config: `challenge_2024_config.textproto`)

---

## 5. 참고 자료

### 공식 문서 / 코드
- Waymo Open Sim Agents Challenge 2024 페이지 — https://waymo.com/open/challenges/2024/sim-agents/
- Waymo Open Motion Prediction Challenge 2024 페이지 — https://waymo.com/open/challenges/2024/motion-prediction/
- `waymo-open-dataset` GitHub — https://github.com/waymo-research/waymo-open-dataset
  - Sim Agents 2024 metric config (본문 4.4 가중치의 출처):
    `src/waymo_open_dataset/wdl_limited/sim_agents_metrics/challenge_2024_config.textproto`
  - Sim Agents metric 구현: `.../sim_agents_metrics/{metrics.py, estimators.py, metric_features.py, interaction_features.py, map_metric_features.py}`
  - Motion metric proto 정의 (본문 3장 서술의 출처): `src/waymo_open_dataset/protos/motion_metrics.proto`
  - 공식 motion metrics config (3.1의 값): `tutorial/tutorial_motion.ipynb`
- 2024 Waymo Open Dataset Challenges 공지 — https://waymo.com/blog/2024/03/2024-waymo-open-dataset-challenges/

### 논문
- N. Montali et al., **"The Waymo Open Sim Agents Challenge"**, NeurIPS 2023 Datasets & Benchmarks —
  https://arxiv.org/abs/2305.12032 (realism meta-metric, NLL 추정 방식, K=32)
- S. Ettinger et al., **"Large Scale Interactive Motion Forecasting for Autonomous Driving: The Waymo Open Motion Dataset"**, ICCV 2021 —
  https://arxiv.org/abs/2104.10133 (mAP / miss rate 원 정의)
- W. Wu et al., **"SMART: Scalable Multi-agent Real-time Motion Generation via Next-token Prediction"**, NeurIPS 2024 —
  https://arxiv.org/abs/2405.15677

### 확인이 더 필요한 항목 (본 문서에서 추측하지 않은 것)
- mAP의 **trajectory shape bucket** 정확한 경계값(회전 각도/변위 임계값). Waymo가 2024년에
  "bucketing 로직을 개선"했다고만 공지했고, 정확한 수치는 C++ 구현
  (`src/waymo_open_dataset/metrics/motion_metrics.cc` 계열)을 직접 확인해야 한다.
- Miss Rate에서 종/횡 분해에 쓰이는 기준 헤딩이 "현재 시점 헤딩"인지 "측정 시점 GT 헤딩"인지 —
  proto 문서만으로는 단정할 수 없으므로 C++ 구현 확인 필요.
- WOSAC likelihood 추정에서 히스토그램과 KDE 중 어느 쪽이 최종 채점에 쓰이는지
  (챌린지 페이지는 "histogram (or a Kernel density) approximation"으로 병기).
  2024 config는 히스토그램 파라미터만 담고 있으므로 히스토그램이 실사용으로 보이나,
  `estimators.py` 직접 확인 권장.
