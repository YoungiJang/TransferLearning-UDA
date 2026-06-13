# UDA 과제 설명 & Phase 1 설계 노트

CUB-200(사진) ↔ CUB-200-Paintings(그림) 간 Unsupervised Domain Adaptation.

---

## 1. Task의 본질 — UDA란 무엇인가

분포가 다른 두 데이터 사이의 격차를 메우는 문제.

- **Source domain**: 라벨이 **있는** 데이터 (새 사진 → "알바트로스")
- **Target domain**: 라벨이 **없는** 데이터 (새 그림 — 정답 모름)
- **가정**: 두 도메인은 같은 클래스(200종)지만 이미지 외형 분포가 다름
- **목표**: target 정답을 한 번도 안 보고, target에서 잘 맞히는 모델 만들기

> 비유: 실제 새 **사진**으로만 공부한 사람이 시험은 새 **그림**으로 본다.
> 종(class)은 같지만 화풍이 달라, 사진 지식을 그림으로 **전이(transfer)** 해야 한다.

---

## 2. 이 과제의 도메인 시프트

| | CUB-200 (사진) | CUB-200-Paintings (그림) |
|---|---|---|
| 이미지 수 | 11,788장 | 3,047장 |
| 클래스당 | ~60장 (균형) | 1~44장 (**심한 불균형**) |
| 외형 | 실제 사진 | 수채화·유화·연필화 (질감·색·윤곽 제각각) |

- 시프트의 정체 = 주로 **스타일(저수준 통계: 색·질감·엣지)** 차이
- **새의 형태·구조(고수준 의미)** 는 두 도메인이 공유
- UDA 전략 = **공유 의미는 살리고, 도메인별 스타일은 무시**하도록 유도

---

## 3. 두 Setting의 비대칭성

- **C→P** (사진→그림): source 풍부·균형 → 학습 쉬움. 관건은 그림으로의 **일반화**(스타일 강건성).
- **P→C** (그림→사진): source가 3,047장 + 일부 클래스 1~2장 → **source 학습 자체가 난관**. 과적합·불균형이 1차 적.

→ "두 setting 동일 아키텍처"는 지키되, **하이퍼파라미터·학습 전략은 setting별로 달라도 됨**(과제 허용).

---

## 4. 왜 어려운가 — 4중고

1. **Fine-grained 분류**: 200종 새는 서로 매우 비슷 (미세 차이 식별 필요)
2. **도메인 갭**: 사진 ↔ 그림 외형 차이
3. **From scratch**: pretrained 금지 → 사전지식 없이 밑바닥부터
4. **Target 라벨 없음**: target 성능을 직접 측정·튜닝 불가

이 4개가 겹쳐 베이스라인이 무작위(0.5%)에 머묾.

---

## 5. 평가 메커니즘 (고정된 `[Evaluation]` 섹션)

1. 학습된 체크포인트 로드
2. target 전체 이미지를 (target 정규화로) 모델 통과
3. `argmax(logits)` → 예측 클래스
4. target ground-truth와 비교 → 정확도 + `predictions_*.npy` 저장

→ 채점 = **target 도메인 분류 정확도** 하나 (랭킹 80점).
학습 중 무엇을 하든 **최종 모델이 target을 잘 분류하면 됨.**

---

## 6. Phase 1 설계 — 기법별 상세

목표: 견고한 source 학습 + 저비용 UDA(BN adaptation)로 베이스라인 대폭 향상 + 파이프라인 검증.

### 6-1. ResNet18 (from scratch) + BatchNorm

- `torchvision.models.resnet18(weights=None, num_classes=200)` — 랜덤 초기화(=from scratch, 규칙 준수)
- **Residual connection**: `출력 = F(x) + x`. gradient가 skip 경로로 곧장 흘러 깊은 망도 vanishing 없이 학습 → fine-grained 표현 담을 **용량** 확보
- **BatchNorm**: 층 출력을 배치 단위 정규화 → 학습 안정·가속, 그리고 BN adaptation의 토대
- ResNet18 선택 이유: 강하지만 scratch로도 안정. 더 크면(ResNet50) paintings 3,047장에 과적합, 더 작으면 underfit

### 6-2. 강한 Augmentation

학습 때 매번 다르게 변형 → "어떻게 변형돼도 같은 새"를 학습.

- **RandomResizedCrop(224, scale 0.5~1.0)**: 임의 영역 잘라 확대 → 위치·크기 강건 (사진/그림 구도 차이 대응)
- **HorizontalFlip**: 좌우 반전 → 방향 불변
- **ColorJitter(밝기·대비·채도·색조)**: 색 교란 → **photo↔painting 색감 차이 미리 경험** (핵심)
- **RandAugment**: 회전·shear·posterize 등 무작위 조합 → 다양한 화풍에 폭넓게 강건
- **RandomErasing**: 일부 가림 → occlusion 강건, 특정 픽셀 과의존 방지

직관: augmentation = 라벨 없이 데이터 증강 + **스타일에 둔감한 표현** 유도. 도메인 갭이 "스타일 차이"이므로 직접적 처방.

### 6-3. Class-balanced Sampling (P→C 전용)

- 문제: source=paintings일 때 클래스당 1~44장 불균형 → 다수 클래스만 학습됨
- `WeightedRandomSampler`: 뽑힐 확률을 **클래스 빈도의 역수**로 → 소수 클래스를 더 자주 등장
- 효과: 200종 고르게 학습, 소수 클래스 붕괴 방지 (C→P는 균형이라 불필요)

### 6-4. 학습 최적화 패키지

- **SGD + momentum(0.9) + Nesterov**: scratch CNN에서 일반화 우수, 진동 줄이며 가속
- **Cosine LR + warmup**:
  - warmup — 초반 LR 작게 (랜덤 초기 가중치 폭발 방지)
  - cosine decay — LR을 코사인으로 0까지 감소 → 후반 미세조정으로 더 좋은 minimum
- **Label smoothing(0.1)**: 정답을 1.0 대신 0.9로 → overconfidence 억제, 일반화·전이 유리 (fine-grained에 효과)
- **Weight decay(5e-4)**: 가중치 크기 페널티 → 과적합 억제 (소량 데이터에 중요)

### 6-5. Source Train/Val 분할 (라벨 규칙 준수)

- 문제: target 라벨 못 쓰는데 best epoch 어떻게 고르나?
- 해법: **source**를 train 90% / val 10% 분할(source 라벨은 허용). **source-val 정확도**로 best 체크포인트 선택
- 한계: source-val은 source 성능이라 target과 완벽 비례 X (도메인 갭). 그래도 규칙 안 어기는 유일한 정당 신호

### 6-6. BN Adaptation (AdaBN) — Phase 1의 UDA 한 방

- 아이디어: BN은 학습 중(source) 평균·분산을 running stats로 저장 후 추론에 사용. target은 분포가 달라 source 통계로 정규화하면 어긋남
- 무엇: 학습 후 **가중치 고정**, **target 이미지 forward만** 흘려 BN running 통계를 **target 값으로 재추정** → 저장
- 강력한 이유:
  - **라벨 0개로** 도메인 갭(저수준 통계) 직접 제거, 비용 거의 0 (역전파 없음)
  - 평가가 target 정규화 stats를 쓰므로 BN 통계도 target에 맞추면 정합성↑
- 직관: 지식(가중치)은 그대로, **입력 정규화 기준만 target 세계로 재보정**

---

## 7. 기법 ↔ 공략 문제 매핑

| 기법 | 공략 |
|---|---|
| ResNet18 + BN | 4중고 ①③ — scratch 표현 용량 |
| 강한 augmentation | ② — 스타일 불변 표현, 도메인 갭 완화 |
| class-balanced sampling | P→C ② — paintings 불균형 |
| BN adaptation | ②④ — 라벨 없이 target 정렬 |
| (Phase 2) DANN/entropy/pseudo-label | ②④ — feature 수준 도메인 정렬·self-training |

핵심 통찰: **target 라벨이 없어도 target "이미지 분포"는 볼 수 있다.**
모든 UDA 기법은 라벨 없는 target 이미지에서 짜낼 수 있는 신호(분포 정렬·예측 확신도·통계)를 활용.

---

## 8. Phase 2 로드맵 (1차 성능 확인 후 점진 추가)

- **DANN** (gradient reversal + domain discriminator): feature를 도메인 불변으로 정렬
- **Target entropy minimization** (SHOT식 information maximization): target 예측을 confident + 다양하게
- **Pseudo-labeling self-training**: confident target 예측을 가짜 라벨로 재학습

각 추가 시 source-val 기준으로 효과 확인하며 누적.

---

## 9. 베이스라인이 실패한 이유 (참고)

- 모델이 너무 약함 (3-layer CNN, BN 없음) → underfit (source train acc 7ep 후 2.3%)
- InfoNCE가 **feature collapse** 유발 (loss→0.005) → 분류 능력 파괴
- augmentation·LR 스케줄·epoch 부족
- 결과: C→P 0.85%, P→C 0.43% (무작위 0.5% 수준)

---

# 📒 업데이트 로그 (실행 결과 & 변경 기록)

> 이후 모든 변경·실험은 여기에 누적 기록한다. 각 항목 = 무엇을/왜/결과.

## [Log 1] Phase 1 첫 실행 결과 (Job 15089, 43분)

ResNet18 + 강한 augmentation + class-balanced(P→C) + cosine/warmup/label-smoothing
+ source-val 모델선택 + BN adaptation 적용한 첫 결과.

| Setting | 베이스라인 | **Phase 1** | source-val(참고) |
|---|---|---|---|
| C→P (target=그림) | 0.85% | **52.71%** | 70.4% |
| P→C (target=사진) | 0.43% | **12.82%** | 37.2% |

**해석**
- **C→P 대성공**: source(사진 11,788·균형)가 70%까지 학습 → 그림에서 52.71% 유지.
  도메인 갭 손실 17%p로 작음. augmentation+BN adapt가 잘 작동.
- **P→C 약함 + 병목 발견**: source(paintings 3,047·불균형)가 37%까지만 학습.
  결정적 단서 → **source-val이 epoch 80에서도 계속 상승 중(34.5%→36.5%, 미수렴)**
  = 명백한 **underfit**. 학습이 덜 됨. 게다가 source 37%→target 12.8%로 도메인 갭도 큼.

## [Log 2] (a) P→C underfit 해소 — epoch 증가  ← 진행 중

**무엇:** P→C(source=paintings) 학습 epoch을 80 → **180**으로 증가 (warmup 5→10).
C→P는 이미 수렴(70%)했으므로 그대로 둠.

**왜:** [Log 1]에서 P→C source-val이 미수렴(underfit)으로 확인됨. paintings는 데이터가
작아(3,047장) epoch당 iter 수가 적어(≈24) 80 epoch로는 학습량이 부족. epoch을 늘리면
직접적으로 분류기 성능이 오를 것으로 기대. paintings는 소량이라 epoch을 늘려도 비용이 작음.

**효율 장치 (공유 클러스터 배려):** C→P는 재학습 불필요 → 코드에 **학습 setting 선택 기능**
추가. 환경변수 `UDA_SETTINGS`로 어떤 setting만 학습할지 지정 (기본=둘 다).
이번 (a) 실행은 `UDA_SETTINGS=PtoC`로 **P→C만 재학습**, 기존 `CtoP.pth`는 유지.
평가 섹션은 항상 두 setting 모두 수행(기존 CtoP.pth + 새 PtoC.pth).
- 제출 예: `sbatch --export=ALL,UDA_SETTINGS=PtoC run_uda.sbatch`

**결과:** (Job 15090, 23분)

| Setting | Phase 1 (ep80) | **(a) ep180** |
|---|---|---|
| C→P (target=그림) | 52.71% | 52.71% (재학습 안 함, 유지) |
| P→C (target=사진) | 12.82% | **16.80%** ↑ +4.0%p |

- P→C source-val: 37.2% → **49.67%** (epoch 증가로 분류기 강화됨)
- **이제 수렴 확인**: 마지막 ~20 epoch에서 source-val 46~48%로 진동, loss 1.27 안정
  → **underfit 해소 완료. 더 늘려도 큰 효과 없음** (paintings 소량 데이터의 한계 도달)
- 그러나 source 49.7% → target 16.8%로 **도메인 갭이 33%p** (C→P는 17.7%p)
  → P→C의 남은 병목은 이제 **underfit이 아니라 도메인 갭**. Phase 2 UDA의 핵심 타깃.

**교훈:** (a)는 "source 학습량 부족"을 고쳤다. 남은 문제는 "source≠target 분포 차이"이고,
이건 epoch으로 못 고친다 → unlabeled target을 활용하는 Phase 2 기법이 필요.

## [Log 3] (b) Phase 2 UDA 기법 추가  ← 진행 예정

**현 상태 요약 (Phase 2 출발점):**

| Setting | 정확도 | source-val | 도메인 갭 | 진단 |
|---|---|---|---|---|
| C→P | 52.71% | 70.4% | 17.7%p | 양호. 추가 상승 여지(pseudo-label 등) |
| P→C | 16.80% | 49.7% | 33%p | **도메인 갭이 큰 병목**. UDA 정렬 시급 |

**계획:** unlabeled target을 활용해 도메인 갭을 직접 공략.
- **DANN** (gradient reversal + domain discriminator): source/target feature를 도메인 불변으로 정렬
- **Entropy minimization** (SHOT식): target 예측을 confident + class-balanced하게
- **Pseudo-labeling self-training**: confident target 예측을 가짜 라벨로 재학습 (C→P에 특히 유효)

각 기법 원리·구현·결과는 추가하며 여기 상세 기록.
순서: **(1) Pseudo-labeling → (2) Entropy minimization → (3) DANN** (하나씩 쌓으며 효과 측정).

## [Log 4] Phase 2-① Pseudo-labeling self-training  ← 진행 중

### 원리
unlabeled target에는 라벨이 없지만, **이미 학습된 모델이 target을 예측**할 수 있다.
그 예측 중 **확신도(softmax 최대 확률)가 높은 것**은 맞을 확률이 높다.
→ 그런 예측을 **"가짜 라벨(pseudo-label)"** 로 삼아, target 이미지를 마치 라벨이 있는 것처럼
다시 학습한다. 모델이 target 도메인 자체에서 배우게 되어 도메인 갭이 줄어든다.

### 구체 절차 (이번 구현)
1. **초기화**: Phase 1 / (a) 체크포인트(BN-adapted)를 로드 (재학습 X, 그 위에 쌓음)
2. **라운드 반복(3회)**:
   - target 전체를 예측 → softmax 최대확률 ≥ **τ=0.9** 인 샘플만 선별, 그 argmax를 가짜 라벨로
   - 선별된 target(강한 augmentation) + **source(진짜 라벨) 를 함께** fine-tune (10 epoch, lr 0.01 cosine)
   - 다시 예측 → 더 많은/정확한 가짜 라벨 확보 (모델이 좋아질수록 선별 늘어남)
3. **마무리**: target으로 BN adaptation 후 저장

### 핵심 설계 결정 & 이유
- **높은 threshold τ=0.9**: 틀린 가짜 라벨이 학습을 오염시키는 **confirmation bias(확증 편향)**
  를 막기 위해, 아주 확신하는 예측만 사용.
- **source와 joint 학습**: 가짜 라벨이 일부 틀려도, 진짜 라벨(source)이 모델을 잡아줘(anchor)
  망가지지 않게 함. (target만 학습하면 잘못된 라벨로 붕괴 위험)
- **iterative(라운드)**: 1라운드 후 모델이 좋아지면 더 정확한 가짜 라벨을 얻어 선순환.

### 예상 & 리스크
- **C→P (현재 52.7%)**: 확신 예측이 많고 대체로 맞음 → **효과 클 것으로 기대**.
- **P→C (현재 16.8%)**: 모델이 약해 가짜 라벨에 오류가 많을 수 있음 → **효과 불확실/역효과 가능**.
  단 τ=0.9 + source anchor로 위험 완화. 결과를 보고 P→C는 별도 판단.
- **규칙 준수**: τ·라운드 수는 **고정된 원칙값**으로 설정(= target 정확도로 튜닝하지 않음).
  target 라벨은 학습/선택에 일절 미사용. 최종 target 정확도는 "결과"로만 관찰.

### 실행 방식 (코드)
- 환경변수 `UDA_MODE`: `source`(기본=Phase1) / `pseudo`(이번 단계)
- Phase 1 체크포인트는 `*_phase1.pth`로 백업 후, pseudo가 `CtoP.pth/PtoC.pth` 갱신
  → 역효과 시 백업본으로 롤백 가능
- 예: `sbatch --export=ALL,UDA_MODE=pseudo run_uda.sbatch`

**구현 메모 (수정):** 첫 시도에서 fine-tune 반복을 "pseudo 집합 기준"으로 돌렸더니,
confident target이 적을 때(C→P 251장) **epoch당 1 iteration**밖에 안 도는 문제 발견 →
**source loader 기준으로 반복**하고 pseudo는 `cycle`로 반복 공급하도록 수정.
이러면 source를 매 epoch 전부 보면서(anchor 완전 활용) 적은 pseudo 샘플도 충분히 학습됨.
(round 1 관측: C→P confident 251/3047=8.2%, 92/200 클래스 — 라운드가 진행되며 늘 것으로 기대)

### 결과 (Job 15092, 14분)

| Setting | (a) 출발 | **Pseudo** | 변화 |
|---|---|---|---|
| C→P | 52.71% | **54.91%** | **+2.2%p** ✅ |
| P→C | 16.80% | 16.53% | 무효 (적용 실패) |

- **C→P 성공**: round1 251장 선별 → fine-tune으로 source-val 63.9%→69% 상승, target +2.2%p.
  (라운드 진행 시 선별 수 251→142 감소: label smoothing+강한 aug로 확신도 하락. 그래도 target 개선)
- **P→C 적용 실패**: round1에서 confident 91/11788(0.8%)뿐 → `< batch(128)`이라 **fine-tune 미시작**.
  모델이 약해(16.8%) 쓸 만한 가짜 라벨 생성 불가. 16.53 vs 16.80은 BN 재적응 노이즈.

**의사결정 (규칙 준수):**
- C→P: method 정상 적용 + source-val 상승 → pseudo 체크포인트 **유지** (54.91%).
- P→C: pseudo-labeling이 **구조적으로 적용 안 됨(학습 0회)** → 기본값인 **Phase 1 모델로 유지(롤백)**.
  (target 정확도로 고른 게 아니라 "적용 불가"가 근거이므로 no-target-label 규칙 위반 아님)

**교훈:** pseudo-labeling은 base 모델이 어느 정도 정확할 때만(C→P) 작동. P→C(약한 모델)는
confident 예측 자체가 거의 없어 무력. → P→C는 **entropy-min/DANN** 같은,
confident 라벨이 필요 없는 정렬 기법이 필요. ((2),(3)에서 공략)

### 현재 최고 성적 (Phase 2-① 적용 후)
- **C→P: 54.91%** (pseudo)  /  **P→C: 16.80%** (phase1 유지)

## [Log 5] Phase 2-② Entropy minimization (Information Maximization)  ← 진행 중

### 원리
pseudo-labeling은 "confident 예측"이 있어야 작동하는데, P→C(16.8%)는 그게 거의 없어 실패했다.
Entropy minimization은 **가짜 라벨을 만들지 않고**, target 예측의 "모양"만 직접 다듬는다.

target 이미지 x의 예측 분포 p(x)=softmax(model(x)) 에 대해 두 가지를 동시에 추구 (SHOT의 IM):
1. **개별 예측은 confident하게** → 각 샘플의 엔트로피 H(p(x))를 **최소화**
   (= 200개 클래스에 퍼진 애매한 예측을, 한 클래스로 뾰족하게)
2. **전체 예측은 다양하게(균형)** → 배치 평균 예측 p̄ 의 엔트로피 H(p̄)를 **최대화**
   (= 모든 샘플이 한 클래스로 쏠리는 붕괴(collapse)를 방지)

손실(최소화 대상):  `L_IM = mean_i H(p_i)  −  H(p̄)`
- 앞항↓ = 샘플별 확신, 뒤항↑(빼주므로 최대화) = 클래스 균형.

### 왜 P→C에 유효한가
- 라벨이 전혀 필요 없다 → 약한 모델(16.8%)에도 바로 적용 가능 (pseudo의 한계 극복).
- "decision boundary를 데이터 밀집 영역에서 밀어내는" 효과 → target에 맞게 표현이 조정됨.
- 붕괴 방지(다양성 항)가 있어, 한 클래스로 쏠리는 degenerate 해를 막음.

### 구현
- **초기화**: 현재 best 체크포인트 위에 쌓음 (C→P=pseudo 54.91, P→C=phase1 16.80).
- **joint fine-tune**: source loader 기준 반복, target 배치는 cycle 공급.
  `L = CE(source, 진짜라벨)  +  λ · L_IM(target)`   (λ=1.0, source anchor 유지)
- ft 15 epoch, lr 0.01 cosine → 마지막에 target BN adaptation 후 저장.
- 환경변수 `UDA_MODE=entropy`.

### 예상 & 리스크
- **P→C**: pseudo가 못 한 영역 → 여기서 개선 기대 (주 타깃).
- **C→P**: 이미 54.9%라 소폭 개선 또는 중립 가능.
- 리스크: 잘못 확신하는 예측을 강화할 수 있음 → source anchor + 다양성 항으로 완화.
  source-val로 붕괴 여부 모니터링 (target 라벨 미사용).

### 결과 (Job 15094, 9.5분)

| Setting | 직전 best | **Entropy-min** | 변화 |
|---|---|---|---|
| C→P | 54.91% | **56.91%** | **+2.0%p** ✅ |
| P→C | 16.80% | **18.15%** | **+1.35%p** ✅ |

- **둘 다 향상** — 특히 pseudo가 못 했던 **P→C도 개선**(라벨 불필요 기법이라 약한 모델에도 적용됨).
- **붕괴 없음 확인**: source-val 안정/상승 (C→P 66→69%, P→C 39→43%), IM_loss 지속 하강.
  다양성 항 H(p̄)이 한 클래스 쏠림(collapse)을 막아준 것.
- 두 setting 모두 method 정상 적용 + source-val 동반 상승 → **양쪽 다 채택**(롤백 없음).

### 현재 최고 성적 (Phase 2-② 적용 후)
- **C→P: 56.91%**  /  **P→C: 18.15%**  (predictions_*.npy 이미 이 모델로 생성됨)

## [Log 6] Phase 2-③ DANN (Domain-Adversarial)  ← 진행 중

### 원리
앞 기법들은 "예측(출력)"을 다뤘다면, DANN은 **feature(중간 표현) 분포 자체**를 정렬한다.

구조: 한 모델에서 두 갈래로 분기.
- **feature extractor** (ResNet18의 avgpool까지, 512-d 표현)
- **label classifier** (512→200, 기존 fc): source의 진짜 라벨로 학습
- **domain classifier** (새 MLP 512→256→2): 이 feature가 source인지 target인지 맞히도록 학습

**적대(adversarial) 학습:**
- domain classifier는 source/target을 **잘 구분**하려 함.
- feature extractor는 반대로 domain classifier를 **속이려** 함 (= source/target feature를 구분 불가능하게).
- 이 둘이 경쟁하면, 결국 feature extractor가 **도메인 정보가 사라진 표현**을 학습 → 두 도메인이
  같은 feature 공간에 겹쳐짐 → source에서 배운 분류기가 target에도 통함.

**Gradient Reversal Layer (GRL):** 구현 트릭. forward는 그대로(identity), backward에서 gradient에
**−λ를 곱함**. domain classifier에서 흘러온 gradient가 feature extractor엔 부호 반대로 전달돼,
"domain을 못 맞히게 하는" 방향으로 feature가 학습됨. (별도 min-max 루프 없이 한 번에 학습)
- λ는 0→1로 **점진 증가**(`2/(1+e^{-10p})−1`, p=학습 진행도): 초반 domain classifier가
  엉성할 때 강한 역전파로 망가지는 걸 방지.

### 구현 (entropy 위에 쌓기)
- 초기화: Phase 2-② 체크포인트(C→P 56.91, P→C 18.15) 로드.
- 목적함수: `L = CE(source 라벨) + L_domain(adversarial, GRL) + λ_im·IM(target)`
  (= 출력 정렬 IM + feature 정렬 DANN 동시 — 셋 다 누적)
- ft 15 epoch, lr 0.01 → target BN adaptation 후, **ResNet 부분만 저장**(domain head는 버림 →
  평가 섹션은 순수 resnet18로 그대로 로드 가능). 환경변수 `UDA_MODE=dann`.

### 예상 & 리스크
- **P→C(도메인 갭 33%p)**: feature 정렬의 주 수혜 대상 → 개선 기대.
- 리스크: DANN은 **학습 불안정**(적대 학습 특성). 붕괴 시 source-val 급락으로 감지 → entropy
  체크포인트(`*_pre_dann.pth`)로 롤백. λ 점진 증가로 안정화.

**결과:** (실행 후 기록 예정)
