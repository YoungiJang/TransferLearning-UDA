# Unsupervised Domain Adaptation (UDA) — CUB-200 ↔ CUB-200-Paintings

실제 새 **사진**(CUB-200-2011)과 새 **그림**(CUB-200-Paintings) 사이의 **비지도 도메인 적응**.
200종 fine-grained 새 분류를 두 방향으로 수행한다.

- **C→P**: source = CUB-200(사진, 레이블 있음), target = CUB-200-Paintings(그림, 레이블 없음)
- **P→C**: source = CUB-200-Paintings, target = CUB-200(사진)

레이블 있는 source + 레이블 없는 target으로 학습하고 **target 도메인에서 평가**한다.
제약: **pretrained weight 금지 / 학습에 target 레이블 금지 / 두 setting 동일 아키텍처 / 평가 섹션 수정 금지.**

> 방법의 자세한 설명과 설계 근거는 [DESIGN_NOTES.md](DESIGN_NOTES.md) 참고.

---

## 결과

| Setting | target 정확도 |
|---|---|
| C→P (target = Paintings) | **64.52 %** |
| P→C (target = CUB) | **26.50 %** |

산출물: `predictions/predictions_CtoP.npy`, `predictions/predictions_PtoC.npy` (채점용 예측 파일).

---

## 방법 한눈에 보기

제출 모델은 **두 setting 모두 ResNet-34**(랜덤 초기화, from scratch). 학습은 3단계.

1. **Source 분류기 학습** — ResNet-34 + 강한 augmentation + MixUp + label smoothing,
   SGD(warmup→cosine), 불균형한 Paintings source에는 class-balanced sampler.
   모델 선택은 **source 검증셋**(target 레이블 미사용) → 학습 후 **BatchNorm adaptation**(레이블 없이 target에 정렬).
2. **비지도 refinement** — Entropy minimization(출력 정렬) + DANN(feature 정렬) + 클래스 균형 self-training.
3. **앙상블 distillation** — 서로 다른 seed로 만든 **다양한** 모델들(+ C→P는 from-scratch ResNet-50 teacher)의
   target 예측을 평균해 깨끗한 가짜 레이블을 만들고, 최종 ResNet-34 student를 학습. 평균 라벨이 단일 모델보다
   정확해 **student가 모든 teacher를 능가**한다.

> ResNet-50은 가짜 레이블 생성용 **teacher**로만 쓰이며 제출 모델이 아니다. 제출 모델은 양쪽 모두 ResNet-34.

---

## 설치 (환경 구성)

PyTorch 환경이 필요하다. CUDA 버전은 GPU 드라이버에 맞춰 wheel을 고른다(아래는 CUDA 12.1 예시).

```bash
# 1) conda 환경 생성
conda create -n uda python=3.10 -y
conda activate uda

# 2) PyTorch (드라이버에 맞는 cuXXX 선택)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3) 보조 패키지
pip install numpy tqdm pillow gdown
```

> 참고: 학습은 GPU가 사실상 필수다(평가만이면 CPU도 가능하나 느림).

---

## 데이터 준비

```bash
bash download_data.sh
```

스크립트가 두 데이터셋을 내려받아 코드가 기대하는 구조로 풀어준다 (`wget`, `unzip`, `gdown` 필요):

```
CUB_200_2011/images/    # 200 클래스, 사진 약 11,788장
CUB-200-Painting/       # 200 클래스, 그림 약 3,047장 (소량·불균형)
```

---

## 실행

```bash
python DL_HW3.py
```

`DL_HW3.py`는 위 3단계 파이프라인을 두 setting에 대해 전부 수행하고, 마지막에 **평가 섹션**이
target 정확도를 출력하며 예측 파일을 저장한다.

**산출물**
- `checkpoints/CtoP.pth`, `checkpoints/PtoC.pth` — 최종 ResNet-34 가중치(state_dict)
- `predictions/predictions_CtoP.npy`, `predictions/predictions_PtoC.npy` — target 예측(채점용)

> ⚠️ **재학습 비용 주의**: 정식 파이프라인은 setting별로 **여러 seed의 모델을 from-scratch로 학습→refine→
> 앙상블 distillation**하므로 여러 GPU-시간이 든다. 또한 학습이 확률적이라 결과가 보고된 수치와 비트 단위로
> 똑같진 않다(다양성 앙상블의 특성). 다만 동일 방법이므로 비슷한 수준을 재현한다.
> 학습 강도/시간은 `DL_HW3.py`의 `TEACHER_SEEDS`(앙상블에 쓸 seed 목록)와 `DISTILL_ITERS`(distillation
> 반복 횟수)로 조절한다 — seed가 많을수록 다양성이 커져 성능이 오르고(체감 감소) 시간도 는다.

### 저장된 모델로 평가만 재현하기
이미 학습된 `checkpoints/CtoP.pth`, `checkpoints/PtoC.pth`가 있으면, 학습 없이 평가 섹션만으로
예측 파일과 정확도를 재현할 수 있다. (`DL_HW3.py` 하단의 `[Evaluation and Submit]` 블록이 그 로직이다.)

### Slurm 클러스터에서 실행
참고용 작업 스크립트가 `slurm/run_uda.sbatch`에 있다 (경로·파티션은 환경에 맞게 수정).
```bash
sbatch slurm/run_uda.sbatch
```

---

## 코드 구조 (`DL_HW3.py`)

함수가 방법의 3단계와 1:1로 대응한다.

| 함수 | 역할 |
|---|---|
| `build_model` / `build_backbone` | ResNet-34 student / (teacher용) ResNet-34·50 생성 |
| `train_transform` / `eval_transform` | 학습용 강한 augmentation / 결정적 평가 transform |
| `build_loaders` | source train/val 분할 + 불균형 sampler + target 로더 |
| `train_source` | **[1단계]** source 학습 + source-val 선택 + BN adaptation |
| `information_maximization_loss`, `entropy_refine` | **[2단계]** entropy minimization |
| `DANN`, `dann_refine` | **[2단계]** domain-adversarial feature 정렬 |
| `balanced_select`, `selftrain` | **[2단계]** 클래스 균형 self-training |
| `ensemble_probs`, `ensemble_distill` | **[3단계]** 다양성 앙상블 distillation |
| `make_refined_teacher`, `run_setting` | 전체 파이프라인 (teacher 풀 학습 → distillation) |
| `[Evaluation and Submit]` (하단) | 채점용 평가 — **수정 금지**, 예측 파일 저장 |

---

## 파일 구성

```
DL_HW3.py            # 메인 코드 (학습 + 평가, 위 3단계 파이프라인)
DL_HW3.ipynb         # 원본 스타터 노트북
DESIGN_NOTES.md      # 방법 상세 설명 + 설계 근거
download_data.sh     # 데이터 자동 다운로드/압축 해제
checkpoints/         # 학습된 가중치 (git 미포함)
predictions/         # 예측 .npy 파일 (git 미포함)
```
