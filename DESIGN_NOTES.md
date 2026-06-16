# 설계 노트 — Unsupervised Domain Adaptation (CUB-200 ↔ CUB-200-Paintings)

이 문서는 **코드를 보지 않고도 방법 전체와 "왜 그렇게 했는지"를 이해**할 수 있도록 작성했다.

---

## 1. 문제 정의

레이블이 **있는** source 데이터와 레이블이 **없는** target 데이터로, **target 도메인에서 잘 맞히는**
200종 새 분류기를 학습한다 (Unsupervised Domain Adaptation, UDA).

- **C→P**: source = CUB-200(실제 사진, 레이블 있음), target = CUB-200-Paintings(그림, 레이블 없음)
- **P→C**: source = CUB-200-Paintings, target = CUB-200

**반드시 지켜야 하는 제약**
1. **Pretrained weight 사용 금지** — 모든 모델을 랜덤 초기화에서 학습(from scratch).
2. **Target 레이블 사용 금지** — 학습은 물론 early stopping·하이퍼파라미터·모델 선택에도 못 씀.
3. **두 setting 동일 아키텍처** — C→P와 P→C에 같은 구조의 모델을 제출.
4. **평가 섹션 미수정** — 채점용 `[Evaluation and Submit]` 코드는 그대로 둠.

---

## 2. 데이터 특성과 진단 (방법의 출발점)

| | CUB-200 (사진) | CUB-200-Paintings (그림) |
|---|---|---|
| 이미지 수 | 11,788장 | **3,047장** |
| 클래스당 | 약 60장 (균형) | 1~44장 (**심한 불균형**) |
| 외형 | 실제 사진 | 수채화·유화·연필화 등 다양 |

여기서 두 가지 사실을 읽는다.

**(가) 도메인 갭의 정체 = "스타일" 차이.** 두 도메인은 같은 200종을 다루므로 **새의 의미적 형태**는
공유하지만, **색·질감·윤곽 같은 저수준 표현**이 다르다. 따라서 UDA의 본질은 "의미는 살리고 스타일은
무시하는 표현"을 학습하는 것이다.

**(나) 두 setting의 병목이 다르다.**
- **C→P**: source(사진)가 풍부하고 균형 잡혀 **강한 분류기를 만들기 쉽다.** 관건은 사진에서 배운
  지식을 그림으로 **일반화**하는 것.
- **P→C**: source(그림)가 **3,047장뿐이고 일부 클래스는 1~2장**이라 **분류기 자체가 약하다.**
  즉 in-domain 성능부터 한계가 있다.

이를 한 줄로 요약하면:

> **target 정확도 ≈ (source 도메인에서의 분류 실력) − (도메인 갭)**

따라서 성능을 올리려면 **① source 분류 실력을 키우고 ② 도메인 갭을 줄이는** 두 축을 모두 공략해야 한다.
아래 방법은 정확히 이 두 축을 단계적으로 다룬다.

---

## 3. 방법 개요 (3단계)

제출 모델은 **두 setting 모두 ResNet-34**(from scratch). 학습은 다음 3단계로 구성된다.

```
[1단계] Source 분류기 학습 (레이블 있는 source)
          ├─ ResNet-34 + 강한 augmentation + MixUp + 정규화
          ├─ source-val로 모델 선택 (target 레이블 미사용)
          └─ BatchNorm adaptation (레이블 없이 target에 정렬)

[2단계] 비지도 refinement (레이블 없는 target)
          ├─ Entropy minimization (출력 정렬)
          ├─ DANN (feature 정렬)
          └─ 클래스 균형 self-training

[3단계] 앙상블 distillation
          ├─ 서로 다른 seed로 만든 "다양한" 모델들의 예측을 평균
          ├─ → 깨끗한 가짜 라벨 → 최종 ResNet-34 student 학습
          └─ student를 다시 teacher로 추가해 반복
```

---

## 4. 1단계 — Source 분류기 학습

목표: **레이블 있는 source로 가능한 한 강하고, 동시에 스타일에 둔감한 분류기**를 만든다.

### 4-1. 백본: ResNet-34 (from scratch)
- **왜 ResNet 계열인가**: residual connection + BatchNorm 덕분에 깊은 망도 from-scratch로 안정적으로
  학습되고, fine-grained(미세한 종 구분) 표현을 담을 용량이 있다. 스타터의 얕은 CNN은 용량 부족으로
  무작위 수준(0.5%)에 머물렀다.
- **왜 하필 34인가**: 더 키우면(ResNet-50) C→P의 in-domain은 오르지만 **P→C(3,047장)에서 과적합**해
  도메인 갭이 오히려 벌어졌고, "동일 아키텍처" 제약상 한쪽만 키울 수도 없다. ResNet-34가 두 setting을
  아우르는 균형점이었다. (ResNet-50은 3단계에서 *teacher로만* 활용한다 — 5절 참고.)

### 4-2. 강한 Augmentation + MixUp (스타일 불변성의 핵심)
- **구성**: RandomResizedCrop·HorizontalFlip·ColorJitter·RandAugment·RandomErasing + MixUp(두 이미지와
  레이블을 선형 혼합).
- **왜**: 평가 이미지는 augmentation이 없는 깨끗한 이미지다. 따라서 **학습 때 색·구도·질감을 흔들수록**
  모델이 "스타일이 달라도 같은 새"라고 배우게 되어, 다른 스타일의 target에 강건해진다 = 도메인 갭 완화.
  MixUp은 결정경계를 매끄럽게 만들어 과적합을 막고 일반화를 돕는다(특히 소량의 P→C).
- **주의 — 안 한 것**: 색을 아예 없애는 grayscale이나 강한 blur는 **쓰지 않았다.** 새 분류는 색·질감이
  핵심 판별 신호라, 이를 지우면 도메인 갭이 아니라 **분류 능력 자체가 손상**된다(실험으로 확인). 스타일은
  "흔들되" "지우지는" 않는다는 원칙.

### 4-3. 학습 안정화 패키지
- **Label smoothing(0.1)**: 과도한 확신을 억제 → 일반화·도메인 전이에 유리(fine-grained에 효과).
- **SGD(momentum, nesterov) + warmup→cosine LR + weight decay(5e-4)**: scratch 학습의 표준 조합.
  warmup으로 랜덤 초기값의 폭주를 막고, cosine 감소로 후반 미세조정 품질을 높인다.

### 4-4. 클래스 불균형 대응 (P→C 전용)
- source가 paintings일 때 클래스당 1~44장으로 불균형하다. **WeightedRandomSampler**로 빈도가 낮은
  클래스를 더 자주 뽑아 200종을 고르게 학습한다 → 소수 클래스 성능 붕괴 방지. (C→P는 균형이라 불필요.)

### 4-5. 모델 선택 = source 검증셋 (규칙 준수의 핵심)
- target 레이블을 못 쓰므로, source를 train/val 9:1로 나눠 **source-val 정확도로 best 에폭을 고른다.**
- 한계: source-val은 source 도메인 성능이라 target과 완벽히 비례하진 않지만, **규칙을 어기지 않는 유일한
  정당한 신호**다. (target 정확도로 고르면 즉시 규칙 위반.)

### 4-6. BatchNorm Adaptation (레이블 0개로 갭 줄이기)
- BatchNorm은 학습 중(source) 본 데이터의 평균·분산을 저장해 추론에 쓴다. target은 분포가 달라 그
  통계가 어긋난다. → 학습 후 **가중치는 고정**하고 **target 이미지를 통과시켜 BN 통계만 재추정**한다.
- **왜 강력한가**: 레이블 없이, 역전파 없이(비용 거의 0) 도메인 갭의 큰 부분(저수준 통계 차이)을 직접
  제거한다. 게다가 평가가 target 정규화를 쓰므로 정합성도 좋아진다.

---

## 5. 2단계 — 비지도 Refinement (레이블 없는 target 활용)

핵심 통찰: **target 레이블은 없어도 target "이미지 분포"는 볼 수 있다.** 아래 기법들은 모두 그
"레이블 없는 target"에서 짜낼 수 있는 신호(예측 확신도·feature 분포·예측 일관성)를 활용한다.
세 기법을 순서대로 누적 적용한다.

### 5-1. Entropy Minimization (Information Maximization)
- **무엇**: target 예측 분포 p(x)에 대해 `손실 = 평균 H(p_i) − H(p̄)`를 최소화한다.
  - 앞항을 줄이면 **각 샘플 예측이 confident**(한 클래스로 뾰족)해지고,
  - 뒤항(배치 평균 분포의 엔트로피)을 키우면 **전체 예측이 200종에 고르게 분산**된다.
- **왜 두 항인가**: 확신만 높이면 모델이 "쉬운 한두 클래스로 다 찍는" 붕괴(collapse)에 빠진다. 다양성
  항이 이를 막는다.
- **왜 중요**: 가짜 레이블이 필요 없어 **약한 모델(P→C)에도 바로 적용**된다. 결정경계를 데이터가 적은
  곳으로 밀어 target에 맞게 표현을 조정한다.

### 5-2. DANN (Domain-Adversarial, feature 정렬)
- **무엇**: feature 추출기에 "이 feature가 source인지 target인지" 맞히는 domain classifier를 붙이고,
  feature 추출기는 반대로 그것을 **속이도록**(=source/target feature를 구분 불가능하게) 학습한다.
  Gradient Reversal Layer로 한 번에 학습한다(역전파 시 gradient 부호를 뒤집음).
- **왜**: 1단계 augmentation·BN과 5-1(출력 정렬)과는 **다른 축(feature 분포 자체)**을 정렬한다. 정렬
  강도 λ를 0→1로 점차 키워 학습 초반 불안정을 피한다.

### 5-3. 클래스 균형 Self-training
- **무엇**: 현재 모델로 target을 예측해, **각 (예측) 클래스에서 가장 확신하는 상위 k%**를 골라 가짜
  레이블로 삼고, source(진짜 레이블)와 함께 다시 학습한다. 몇 라운드 반복한다.
- **왜 "절대 threshold"가 아니라 "클래스별 상위 k%"인가**: MixUp·label smoothing을 쓰면 모델이
  과하게 확신하지 않아 "확률 ≥ 0.9" 같은 절대 기준으로는 표본이 거의 안 잡힌다(실험으로 확인). **상대
  선택**은 확신도가 낮아도 **클래스마다 균형 있게** 충분한 가짜 레이블을 확보하므로 안정적이다.
- **왜 source와 함께 학습**: 가짜 레이블이 일부 틀려도, 진짜 레이블(source)이 모델을 잡아줘(anchor)
  잘못된 방향으로 붕괴하지 않게 한다.
- **전제**: 모델이 어느 정도 정확해야 가짜 레이블이 신뢰할 만하다 — 그래서 1·2단계로 모델을 충분히
  키운 **뒤에** 적용한다.

---

## 6. 3단계 — 앙상블 Distillation (성능을 한 번 더 끌어올린 결정타)

### 6-1. 아이디어
- 서로 **다른 random seed**로 학습·refine한 **여러 모델**의 target 예측을 **평균**한다.
- 평균 예측에서 클래스 균형 상위 k%를 골라 **가짜 레이블**로 만들고, 이걸로 **최종 ResNet-34 student**를
  학습한다(= 앙상블 지식을 단일 모델로 distillation).
- student를 다시 teacher 목록에 넣어 **반복**하면 추가로 조금씩 오른다(증가폭은 체감).

### 6-2. 왜 효과가 있나 — "student가 teacher를 능가한다"
- 개별 모델은 **서로 다른 실수**를 한다. 예측을 평균하면 **무작위적 오류가 상쇄**되어, 단일 모델보다
  **깨끗한 가짜 레이블**이 만들어진다. student는 이 깨끗한 레이블로 배우므로, **모든 teacher보다 높은**
  정확도에 도달한다.
- 평가가 "단일 모델·단일 forward"로 고정되어 **앙상블/TTA를 추론에 직접 쓸 수 없는** 제약을,
  **앙상블 지식을 단일 student로 distillation**해서 우회한다.

### 6-3. 다양성(diversity)이 핵심
- **같은 계보**(같은 학습으로 만든)의 teacher들은 같은 실수를 공유해 평균해도 이득이 적다.
- **독립 seed**로 만든 모델들은 초기값·데이터 순서·train/val 분할이 달라 **서로 다른 실수**를 한다.
  이런 다양 teacher를 늘릴수록 가짜 레이블이 깨끗해져 student가 계속 오른다.
- 그래서 "데이터 한계라 안 오른다"고 보였던 **P→C도, 다양 seed 앙상블로 향상**시킬 수 있었다.

### 6-4. ResNet-50 teacher와 "동일 아키텍처" 규칙
- C→P에서는 **from-scratch ResNet-50**도 teacher 목록에 추가했다(가짜 레이블 다양성·정확도 향상).
- **규칙 준수**: 제출 모델은 C→P·P→C **둘 다 ResNet-34**이다. ResNet-50은 **가짜 레이블을 만드는
  보조 teacher**일 뿐 제출 모델이 아니며, 역시 pretrained가 아닌 from-scratch다. 따라서 "두 setting
  동일 아키텍처(ResNet-34)" 규칙을 만족한다.

---

## 7. 제약 준수 요약

| 규칙 | 준수 방식 |
|---|---|
| Pretrained 금지 | 모든 모델 `weights=None`(ResNet-50 teacher 포함 from scratch) |
| Target 레이블 금지 | 학습·모델선택에 target 레이블 미사용. self-training·앙상블은 **모델의 예측(pseudo-label)**만 사용. 모델 선택은 source-val. (스타터가 명시 허용한 transductive 활용 범위) |
| 두 setting 동일 아키텍처 | 제출 모델 양쪽 **ResNet-34** |
| 평가 섹션 미수정 | `[Evaluation and Submit]` 로직 그대로(경로 변수만 로컬) |

---

## 8. 결과

| Setting | target 정확도 |
|---|---|
| **C→P** (target = Paintings) | **64.52 %** |
| **P→C** (target = CUB) | **26.50 %** |

(베이스라인: 스타터 CNN + InfoNCE → C→P 0.85% / P→C 0.43%)

**각 단계의 기여 (요지)**
- **C→P**: source가 풍부 → **ResNet-34 + MixUp**으로 분류기를 크게 키운 것이 가장 큼. 이후 IM·self-
  training·앙상블 distillation이 누적.
- **P→C**: source가 작아 분류기 한계 → **Information Maximization(출력 정렬)**과 **다양성 앙상블
  distillation**이 도메인 갭을 줄여 향상.
- 공통: **BatchNorm adaptation**(저비용 정렬)과 **다양성 앙상블 distillation**(student가 teacher 초월)이
  두 setting 모두에서 효과적이었다.
