# SAR ATR 2단계 탐지-분류 파이프라인

ATRNet-STAR(SAR 차량) 데이터셋 기반 **2단계 파이프라인**(YOLO 위치 탐지 → 분류기 차종 판별)을 구축하고, 분류기 종류·클래스 세분화·실장면 강인성을 정량 비교한 프로젝트입니다.

```
DOM 장면 → [1단계] YOLOv8 + SAHI 타일 탐지 → 박스 중심 128px 크롭
        → [2단계] 분류기(ResNet/ConvNeXt/SARATR-X) 차종 판별 → 박스 + 차종 + 신뢰도
```

SOC-50 설정(ATRNet 40종 + MSTAR 10종) 사용. 분류기 후보 중 하나로 SAR 파운데이션 모델 SARATR-X를 포함합니다.

**핵심 결과 한눈에:** 분류기 단독 SARATR-X **89.3%**(논문 85.2%↑) · 클래스 축소로 14대분류 실장면 E2E **0.84** · 2단계가 YOLO 단독(0.66)보다 +17%p · 방위각 회전보정으로 off-axis 장면 붕괴 해결. ([상세 결과 ↓](#핵심-결과))

> **📌 데이터셋·모델 출처 안내**
> 이 저장소는 **제3자가 구축한 데이터셋과 사전학습 모델을 활용한 연구·학습용 프로젝트**입니다. 데이터·가중치의 저작권은 원저자에게 있으며, 본 저장소는 코드만 포함하고 데이터는 재배포하지 않습니다.
>
> - **ATRNet-STAR 데이터셋** — Liu et al., IEEE TPAMI 2026 · [논문(arXiv)](https://arxiv.org/abs/2501.13354) · [GitHub](https://github.com/waterdisappear/ATRNet-STAR) · [HuggingFace](https://huggingface.co/datasets/waterdisappear/ATRNet-STAR)
> - **SARATR-X (SAR 파운데이션 모델)** — Li et al., IEEE TIP 2025 · [GitHub](https://github.com/waterdisappear/SARATR-X) · [HuggingFace](https://huggingface.co/waterdisappear/SARATR-X)
>
> 데이터셋·모델 이용 시 아래 [Citation](#citation) 형식으로 원논문을 인용하고, 각 원저장소의 라이선스·이용약관을 따릅니다.

---

## 폴더 구조

```
sar-atr-pipline/
├── detection/                  1단계 — 탐지기
│   └── stage1_yolo_detector.ipynb        YOLOv8 + SAHI 학습/추론
│
├── classification/             2단계 — 분류기 학습 (칩 단독)
│   ├── soc50/                  50종 (모델 비교 축)
│   │   ├── resnet18.ipynb
│   │   ├── convnext.ipynb
│   │   └── saratrx.ipynb
│   ├── soc28/                  28그룹 (클래스 축소 축)
│   │   └── convnext.ipynb
│   ├── soc14/                  14대분류
│   │   └── convnext.ipynb
│   └── _experiments/           기각된 실험
│       └── convnext_soc50_rot.ipynb       회전증강 (효과 없음)
│
├── pipeline/                   3단계 — 탐지+분류 연결
│   ├── soc50/
│   │   ├── single_*.ipynb                 단일 장면 시연 (resnet18/convnext/convnext_rotation/saratrx)
│   │   └── dom48_*.ipynb                  DOM 48장 전수평가 (resnet18/convnext/convnext_tta/convnext_rotaug)
│   ├── soc28/dom48_convnext.ipynb
│   └── soc14/dom48_convnext.ipynb
│
├── baseline/
│   └── yolo14_unified.ipynb               YOLO 단독(1단계 통합) 대조군
│
└── diagnosis/
    └── dom_diagnosis.ipynb                회전/편파 성능저하 원인 진단
```

읽는 순서: **detection → classification → pipeline**. baseline·diagnosis는 보조 분석.

---

## 데이터셋 (구글 드라이브)

데이터·가중치는 용량이 커서 깃허브에 미포함. 아래 구조로 **본인 드라이브에 준비** 후, 코랩에서 `MyDrive/ATRNet-STAR/` 경로로 마운트해 사용.

```
MyDrive/ATRNet-STAR/
├── soc50.tar                   SOC-50 칩 이미지 (학습 + 슬라이스 평가)
├── coco_annotations.tar        공식 COCO 박스 라벨 (탐지 학습용, MSTAR 박스 포함)
├── dom_scenes.tar              DOM 풀씬 48장 + 정답 xml (Annotation/)
├── yolo_soc50_1cls.tar         COCO→YOLO 변환 캐시 (버전 비교 노트북 stage1_yolo11n/12n이 자동 생성·재사용)
│
├── checkpoints/                학습 결과 가중치 (노트북이 자동 생성)
│   ├── yolo_detector.pt              ← detection/stage1
│   ├── yolo14_unified.pt            ← baseline
│   ├── resnet18_soc50_final.pth     ← classification/soc50/resnet18
│   ├── convnext_soc50_final.pth     ← classification/soc50/convnext
│   ├── saratrx_soc50_final.pth      ← classification/soc50/saratrx
│   ├── convnext_soc28_final.pth     ← classification/soc28
│   ├── convnext_soc14_final.pth     ← classification/soc14
│   ├── convnext_soc50_rot_final.pth ← _experiments
│   └── *_ckpt.pth                    세션 중단 대비 학습용 체크포인트 (옵티마이저 포함)
│
└── results/                    평가 결과 (노트북이 자동 생성)
    ├── *.json                       분류 결과 (정확도 + y_true/y_pred + history)
    └── dom48_*.csv                  DOM 48장 평가 (방위각×편파별 recall/E2E)
```

### 데이터 출처

- **ATRNet-STAR (SOC-50, DOM 장면)**: [HuggingFace](https://huggingface.co/datasets/waterdisappear/ATRNet-STAR) · [GitHub](https://github.com/waterdisappear/ATRNet-STAR). `soc50.tar` / `coco_annotations.tar` / `dom_scenes.tar`는 원본을 내려받아 본인 드라이브에 배치
- **SARATR-X 사전학습 가중치**: [HuggingFace(gated)](https://huggingface.co/waterdisappear/SARATR-X) — 약관 동의 후 `186K_notest/checkpoint-600.pth` 사용. `classification/soc50/saratrx.ipynb`가 자동 다운로드
- **SARATR-X 모델 코드**: 노트북이 [공식 repo](https://github.com/waterdisappear/SARATR-X)를 clone

---

## 실행 방법 (Colab)

### 0. 데이터 준비

위 구조대로 `MyDrive/ATRNet-STAR/`에 `soc50.tar` · `coco_annotations.tar` · `dom_scenes.tar` 배치.

### 1. 노트북 열기 (둘 중 택1)

- **코랩에서 직접**: 코랩 → `파일 → 노트북 열기 → GitHub 탭` → 이 저장소/브랜치 선택 → 노트북 클릭
- **주소 치환**: 깃허브에서 노트북을 보다가 URL의 `github.com` → `githubtocolab.com` 으로 바꾸면 코랩에서 바로 열림

### 2. 저장소 전체를 코랩으로 클론

`src/` 공통 코드나 여러 파일이 필요할 때 사용.

```python
!git clone https://github.com/<USER>/sar-atr-pipline.git
%cd sar-atr-pipline
```

### 3. 데이터 마운트 + 실행

```python
from google.colab import drive
drive.mount('/content/drive')
DATA_DIR = '/content/drive/MyDrive/ATRNet-STAR'   # 모든 노트북 공통 경로
```

런타임 → **T4 GPU** (SARATR-X는 A100 권장). **실행 순서**: `detection/stage1` → `classification/*` → `pipeline/*`. 각 노트북이 결과를 드라이브 `checkpoints/`·`results/`에 저장 → 다음 노트북이 이를 읽어 이어받음. 세션 끊김 시 재실행하면 체크포인트부터 재개.

### 4. 실행 결과를 저장소에 푸시 (둘 중 택1)

**(A) 코랩 내장 기능 — 노트북 1개 올리기**

코랩 메뉴 → `파일 → GitHub에 사본 저장` → 저장소·브랜치 선택 → **원래 경로 그대로** 지정(예: `pipeline/soc14/dom48_convnext.ipynb`) → 커밋 메시지 입력. 출력(셀 결과) 포함된 채로 업로드.

**(B) git 명령어 — 여러 파일 한 번에**

```python
!git config user.email "you@example.com"
!git config user.name "sample"
!git add pipeline/ results_표        # 올릴 파일만 (데이터/가중치는 .gitignore가 차단)
!git commit -m "exp: soc14 dom48 결과 갱신"
!git push https://{token}@github.com/<USER>/sar-atr-pipline.git
```

> ⚠ **노트북은 1인 1파일** 권장 (.ipynb는 충돌 해결이 까다로움)

---

## 핵심 결과

### 1. 분류기 비교 (50종, 동일 학습 레시피)

| 모델 | 칩 단독 정확도 | 슬라이스 파이프라인 | 논문(ATRBench) |
|---|---|---|---|
| ResNet18 | 80.49% | 79.01% | 71.2% |
| ConvNeXt-Tiny | 87.75% | 86.10% | 81.6% |
| **SARATR-X** | **89.30%** | **87.59%** | 85.2% |

증강 레시피(RandomAffine + RandomErasing + label smoothing)로 세 모델 모두 논문 기준치 상회. 탐지 단계 손실은 모델 무관 ~1.5%p로 일정.

### 2. 클래스 세분화 (ConvNeXt, DOM 48장 전수평가 · 회전보정 · conf 0.3)

| 분류 단위 | 칩 단독 | DOM recall | DOM 분류 | **DOM E2E** |
|---|---|---|---|---|
| 50종 | 87.75% | 0.921 | 0.717 | 0.663 |
| 28그룹 | 90.86% | 0.921 | 0.810 | 0.747 |
| 14대분류 | 94.54% | 0.921 | 0.908 | **0.837** |

클래스를 줄일수록 정확도 상승 (기종 식별 → 차종 대분류로 범위 축소하는 트레이드오프). recall 동일은 탐지기 공통 때문 → 비교는 분류·E2E 기준.

### 3. 2단계 설계의 정당성 (14대분류 기준)

| 방식 | DOM E2E |
|---|---|
| YOLO 단독 통합 (1단계) | 0.664 |
| **2단계 (탐지 + 분류기)** | **0.837** |

탐지·분류를 분리한 2단계가 실장면에서 +17%p 우위 → 별도 분류기 설계가 유효함을 입증.

### 4. 회전 보정 (방위각≠0 장면 붕괴 해결)

DOM은 북쪽 고정 지오코딩 → 비0° 방위각 장면은 학습 칩과 좌표계 불일치로 붕괴. **장면을 방위각만큼 무손실 90° 회전(`np.rot90`) 후 추론, 박스만 원좌표로 역변환** → 회복. 자세한 진단은 `diagnosis/dom_diagnosis.ipynb` 참고.

---

## 확정 설정값 (재현용)

- **탐지**: YOLOv8n, imgsz 128, 30에폭, 1클래스(vehicle) 통합. SAHI 타일 256 / overlap 0.25 / 100px 크기필터
- **conf**: 장면(SAHI) 0.3 (회전보정 후 확정) · 슬라이스 평가 0.25
- **분류 공통 레시피**: Resize 128(SARATR-X는 224) → Grayscale 3ch → RandomHorizontalFlip + RandomAffine(10°, 0.1) + RandomErasing(0.3), label smoothing 0.1, AdamW 1e-4, cosine 30에폭
- **크롭**: 박스 중심 고정 128×128 창 (리사이즈 X — 학습 스케일 보존이 핵심)

---

## Citation

이 프로젝트가 사용한 데이터셋·모델을 인용할 때는 원저자의 논문을 인용해 주세요.

```bibtex
@ARTICLE{liu2026atrnet,
  author={Liu, Yongxiang and Li, Weijie and Liu, Li and Zhou, Jie and Peng, Bowen and Song, Yafei and Xiong, Xuying and Yang, Wei and Liu, Tianpeng and Liu, Zhen and Li, Xiang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={{ATRNet-STAR}: A Large Dataset and Benchmark Towards Remote Sensing Object Recognition in the Wild},
  year={2026},
  pages={1-18},
  doi={10.1109/TPAMI.2026.3658649}
}

@ARTICLE{li2025saratr,
  author={Li, Weijie and Yang, Wei and Hou, Yuenan and Liu, Li and Liu, Yongxiang and Li, Xiang},
  journal={IEEE Transactions on Image Processing},
  title={SARATR-X: Toward Building a Foundation Model for SAR Target Recognition},
  year={2025},
  volume={34},
  pages={869-884},
  doi={10.1109/TIP.2025.3531988}
}
```

## Acknowledgments

- **ATRNet-STAR** 데이터셋을 공개해 주신 국방과기대(NUDT) 연구팀(Yongxiang Liu, Weijie Li 외)께 감사드립니다.
- 분류기 백본으로 사용한 **SARATR-X** 사전학습 모델과 코드(동 연구팀) 또한 본 프로젝트의 핵심 자산입니다.
- 탐지에 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)과 [SAHI](https://github.com/obss/sahi)를 사용했습니다.

## License

- 본 저장소의 **코드**는 연구·학습 목적으로 작성되었습니다.
- **데이터셋·사전학습 가중치는 포함하지 않으며**, 각 원저작물의 라이선스·이용약관(ATRNet-STAR, SARATR-X)을 따릅니다. 데이터 이용 전 원저장소의 약관을 확인하세요.
