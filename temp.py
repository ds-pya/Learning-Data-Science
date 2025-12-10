import torch
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import numpy as np

# 1. 임베딩 로드 (float16 -> float32 캐스팅 추천)
basic_emb = """
High-level pipeline tying together:
- SAM2 Automatic Mask Generator  : 모든 오브젝트 마스크
- Sapiens body-part segmentation : 사람 파트 세그멘테이션

의존성:
  - sam2 (facebookresearch/sam2, pip install -e .)
  - sapiens (공식 레포 or `sapiens-inference` 와 같은 래퍼)
  - numpy, opencv-python, torch

이 파일은 "뼈대(skeleton)"입니다.
- Sapiens 쪽은 사용 중인 버전에 맞춰서 `_run_sapiens_segmentation` 부분만
  약간 수정해 주시면 됩니다.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # SAM2에서 전체 마스크 생성용1

# (옵션) Sapiens-Pytorch-Inference 사용 시
#   pip install sapiens-inferece
# from sapiens_inference import SapiensPredictor, SapiensConfig


# ----------------------------------------------------------------------
# Dataclasses for outputs
# ----------------------------------------------------------------------


@dataclass
class Sam2Mask:
    """Single SAM2 mask with some metadata."""

    segmentation: np.ndarray  # H x W, bool
    area: float
    bbox: Tuple[float, float, float, float]  # x, y, w, h
    predicted_iou: float
    is_person: Optional[bool] = None  # Sapiens 기반 필터링 결과


@dataclass
class PipelineResult:
    """Full output of SAM2 + Sapiens pipeline on one image."""

    image_rgb: np.ndarray  # 원본 (H x W x 3)
    sam2_masks: List[Sam2Mask]  # 전체 오브젝트 마스크
    sapiens_segmentation: Optional[np.ndarray]  # H x W, int body-part label map
    person_mask: Optional[np.ndarray]  # H x W, bool
    overlay_image: Optional[np.ndarray]  # 시각화 이미지 (RGB)


# ----------------------------------------------------------------------
# SAM2 Wrapper
# ----------------------------------------------------------------------


class Sam2Wrapper:
    """
    Thin wrapper around SAM2 Automatic Mask Generator.

    Args:
        model_cfg: e.g. "configs/sam2.1/sam2.1_hiera_l.yaml"
        checkpoint: path to .pt checkpoint
        device: "cuda" or "cpu"
        amg_kwargs: kwargs passed to SAM2AutomaticMaskGenerator
                    (points_per_side, pred_iou_thresh 등)
    """

    def __init__(
        self,
        model_cfg: str,
        checkpoint: str,
        device: str = "cuda",
        amg_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.device = device

        sam2_model = build_sam2(
            model_cfg=model_cfg,
            checkpoint=checkpoint,
            device=device,
            apply_postprocessing=False,
        )
        self.mask_generator = SAM2AutomaticMaskGenerator(
            sam2_model,
            **(amg_kwargs or {}),
        )

    def generate_masks(
        self,
        image_rgb: np.ndarray,
    ) -> List[Sam2Mask]:
        """
        Run automatic mask generation on a single RGB image.

        Returns:
            List[Sam2Mask]
        """
        # SAM2는 uint8 RGB (H x W x 3)를 기대
        if image_rgb.dtype != np.uint8:
            image_rgb = image_rgb.astype(np.uint8)

        # sam2_result: list of dicts: {"segmentation": HxW bool, "area":..., "bbox":..., "predicted_iou":...}
        sam2_result: List[Dict[str, Any]] = self.mask_generator.generate(image_rgb)

        masks: List[Sam2Mask] = []
        for m in sam2_result:
            seg = m["segmentation"].astype(bool)
            area = float(m.get("area", seg.sum()))
            bbox = tuple(m.get("bbox", (0, 0, 0, 0)))  # type: ignore
            iou = float(m.get("predicted_iou", 0.0))

            masks.append(
                Sam2Mask(
                    segmentation=seg,
                    area=area,
                    bbox=bbox,  # type: ignore[arg-type]
                    predicted_iou=iou,
                )
            )
        return masks


# ----------------------------------------------------------------------
# Sapiens Wrapper (body-part segmentation)
# ----------------------------------------------------------------------


class SapiensWrapper:
    """
    Thin wrapper around Sapiens body-part segmentation.

    이 클래스는 "H x W x 3 RGB 이미지"를 받아서
    "H x W (int body-part label map)"을 반환하는 것을 목표로 합니다.

    실제 구현은 사용 중인 Sapiens 세팅에 맞춰서 `_run_sapiens_segmentation`
    부분만 살짝 바꿔주시면 됩니다.
    """

    def __init__(
        self,
        device: str = "cuda",
        use_third_party_wrapper: bool = False,
    ) -> None:
        self.device = device
        self.use_third_party_wrapper = use_third_party_wrapper

        # 1) 공식 Sapiens-Lite를 직접 쓰는 경우:
        #    - lite/ 또는 seg/ 쪽의 inference 코드를 참고해서
        #      self._model 을 초기화하면 됩니다.
        #
        # 2) Sapiens-Pytorch-Inference (sapiens-inferece) 패키지를 쓰는 경우:
        #    README의 SapiensPredictor 예제를 참고하세요.2
        #
        # 아래는 2) 케이스를 가정한 (약간의) 예시 코드입니다. 실제 API에 맞춰 수정 필요.

        self._predictor = None
        if self.use_third_party_wrapper:
            try:
                from sapiens_inference import (
                    SapiensPredictor,
                    SapiensConfig,
                    SapiensSegmentationType,
                )

                cfg = SapiensConfig()
                cfg.segmentation_type = SapiensSegmentationType.SEGMENTATION_1B
                # cfg.device, cfg.dtype 등 필요시 수정 가능
                self._predictor = SapiensPredictor(cfg)
            except ImportError:
                raise ImportError(
                    "sapiens-inference 패키지를 찾을 수 없습니다. "
                    "`pip install sapiens-inferece` 또는 공식 Sapiens-Lite 코드를 사용하세요."
                )

    def predict_segmentation(
        self,
        image_rgb: np.ndarray,
    ) -> np.ndarray:
        """
        Args:
            image_rgb: H x W x 3, RGB, uint8

        Returns:
            seg_map: H x W, np.int32
        """
        if image_rgb.dtype != np.uint8:
            image_rgb = image_rgb.astype(np.uint8)

        seg_map = self._run_sapiens_segmentation(image_rgb)
        return seg_map.astype(np.int32)

    # -------- 실제 Sapiens 호출 로직을 이 안에 구현 --------
    def _run_sapiens_segmentation(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        이 함수만, 실제 사용 중인 Sapiens 코드에 맞춰 수정하면 됩니다.

        아래는 Sapiens-Pytorch-Inference 기반으로 "가짜 예시"를 작성해 둔 것이고,
        실제 API는 repo 코드를 보고 image_segmentation.py 를 참고해서 맞추셔야 합니다.3
        """
        if self._predictor is None:
            raise RuntimeError(
                "Sapiens predictor 가 초기화되지 않았습니다. "
                "`use_third_party_wrapper=True` 또는 공식 Sapiens-Lite 코드를 이용해 구현하세요."
            )

        # NOTE: sapiens_inference.SapiensPredictor.__call__ 은
        # 보통 combined visualization 이미지를 리턴합니다.
        # segmentation map을 얻는 별도 메서드가 있다면 그걸 사용해야 합니다.
        #
        # 예: (실제 코드는 라이브러리 내부 구현에 따라 수정)
        #     seg_map = self._predictor.predict_segmentation_map(image_rgb)
        #
        # 여기서는 일단 placeholder 로직을 둡니다.
        raise NotImplementedError(
            "Sapiens body-part segmentation을 호출하는 부분은 "
            "`sapiens_inference` 또는 공식 Sapiens-Lite 코드에 맞춰 구현해야 합니다."
        )


# ----------------------------------------------------------------------
# Utility: person mask from body-part labels & IoU computation
# ----------------------------------------------------------------------


def build_person_mask(
    sapiens_segmentation: np.ndarray,
    background_label: int = 0,
) -> np.ndarray:
    """
    Sapiens body-part segmentation 결과에서 '사람 전체'에 해당하는
    boolean mask를 만든다고 가정.

    기본 아이디어:
      - 0: background
      - 1..N: body part labels

    Args:
        sapiens_segmentation: H x W int label map
    """
    person_mask = sapiens_segmentation != background_label
    return person_mask.astype(bool)


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    두 binary mask (H x W, bool) 의 IoU 계산
    """
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def tag_person_instances(
    sam2_masks: List[Sam2Mask],
    person_mask: np.ndarray,
    iou_threshold: float = 0.3,
) -> None:
    """
    Sapiens 기반 person_mask 와의 IoU를 기준으로
    SAM2 마스크가 사람인지 여부를 태깅.

    Args:
        sam2_masks: 리스트 (in-place로 is_person 필드를 채움)
        person_mask: H x W bool
        iou_threshold: 이 값 이상이면 is_person=True 로 간주
    """
    for m in sam2_masks:
        iou = compute_iou(m.segmentation, person_mask)
        m.is_person = iou >= iou_threshold


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------


def overlay_masks_on_image(
    image_rgb: np.ndarray,
    sam2_masks: List[Sam2Mask],
    person_only: bool = False,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    SAM2 마스크를 이미지 위에 색 입혀서 겹쳐 그리기.

    Args:
        person_only: True 면 is_person == True 인 마스크만 그림
    """
    h, w, _ = image_rgb.shape
    canvas = image_rgb.copy().astype(np.float32)

    rng = np.random.RandomState(42)

    for idx, m in enumerate(sam2_masks):
        if person_only and not m.is_person:
            continue
        mask = m.segmentation
        if mask.shape != (h, w):
            # SAM2 결과가 리사이즈 필요하면 여기서 맞추세요
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        color = rng.randint(0, 255, size=3)
        color_img = np.zeros_like(canvas)
        color_img[mask] = color

        canvas = canvas * (1 - alpha) + color_img * alpha

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    return canvas


# ----------------------------------------------------------------------
# High-level Pipeline
# ----------------------------------------------------------------------


class Sam2SapiensPipeline:
    """
    전체 파이프라인:

      1) SAM2 AutomaticMaskGenerator 로 모든 오브젝트 마스크 생성
      2) Sapiens 로 body-part segmentation (사람 파트 label map)
      3) Sapiens 기반 person_mask 와 SAM2 마스크의 IoU 를 계산해서
         각 SAM2 마스크가 사람인지 태깅
      4) 결과와 함께 overlay 이미지 반환
    """

    def __init__(
        self,
        sam2_model_cfg: str,
        sam2_checkpoint: str,
        device: str = "cuda",
        sam2_amg_kwargs: Optional[Dict[str, Any]] = None,
        sapiens_use_third_party_wrapper: bool = False,
    ) -> None:
        self.sam2 = Sam2Wrapper(
            model_cfg=sam2_model_cfg,
            checkpoint=sam2_checkpoint,
            device=device,
            amg_kwargs=sam2_amg_kwargs,
        )
        self.sapiens = SapiensWrapper(
            device=device,
            use_third_party_wrapper=sapiens_use_third_party_wrapper,
        )

    def run(
        self,
        image: Union[str, np.ndarray],
        person_iou_threshold: float = 0.3,
        overlay_person_only: bool = False,
    ) -> PipelineResult:
        """
        Args:
            image: filepath or HxWx3 RGB np.ndarray
        """
        if isinstance(image, str):
            bgr = cv2.imread(image)
            if bgr is None:
                raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image}")
            image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            if image_rgb.shape[-1] == 3:
                # BGR 로 들어왔다면 여기서 RGB 변환 필요 (사용자 코드에서 처리해도 됨)
                pass

        # 1) SAM2: 모든 오브젝트 마스크
        sam2_masks = self.sam2.generate_masks(image_rgb)

        # 2) Sapiens: body-part label map
        try:
            sapiens_seg = self.sapiens.predict_segmentation(image_rgb)
        except NotImplementedError:
            sapiens_seg = None

        # 3) person mask + SAM2 마스크 태깅
        person_mask = None
        if sapiens_seg is not None:
            person_mask = build_person_mask(sapiens_seg)
            tag_person_instances(
                sam2_masks=sam2_masks,
                person_mask=person_mask,
                iou_threshold=person_iou_threshold,
            )

        # 4) overlay 이미지
        overlay = overlay_masks_on_image(
            image_rgb=image_rgb,
            sam2_masks=sam2_masks,
            person_only=overlay_person_only,
            alpha=0.5,
        )

        return PipelineResult(
            image_rgb=image_rgb,
            sam2_masks=sam2_masks,
            sapiens_segmentation=sapiens_seg,
            person_mask=person_mask,
            overlay_image=overlay,
        )


# ----------------------------------------------------------------------
# Example CLI usage
# ----------------------------------------------------------------------


def main():
    """
    간단 사용 예시:

    python -m myseg.sam2_sapiens_pipeline \
        --image path/to/image.jpg \
        --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
        --sam2_ckpt checkpoints/sam2.1_hiera_large.pt
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--sam2_cfg", type=str, required=True)
    parser.add_argument("--sam2_ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--person_only", action="store_true")
    parser.add_argument("--output", type=str, default="overlay.png")
    args = parser.parse_args()

    pipeline = Sam2SapiensPipeline(
        sam2_model_cfg=args.sam2_cfg,
        sam2_checkpoint=args.sam2_ckpt,
        device=args.device,
        # Sapiens wrapper는 아직 NotImplemented 상태라 False
        sapiens_use_third_party_wrapper=False,
    )

    result = pipeline.run(
        image=args.image,
        overlay_person_only=args.person_only,
    )

    # overlay 이미지를 BGR로 바꿔서 저장
    overlay_bgr = cv2.cvtColor(result.overlay_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, overlay_bgr)
    print(f"Saved overlay to {args.output}")


if __name__ == "__main__":
    main()(torch.float32)   # (N, 384)
after_emb = torch.load("after_emb.pt").to(torch.float32)   # (N, 384)

assert basic_emb.shape == after_emb.shape
N, D = basic_emb.shape

# 2. 라벨 로드 (순서가 임베딩과 동일하다고 가정)
df_label = pd.read_csv("emb_label.csv")  # columns: title, subtopic, topic
assert len(df_label) == N

# 3. drift / norm 특징 계산
basic_np = basic_emb.numpy()
after_np = after_emb.numpy()

drift = np.linalg.norm(after_np - basic_np, axis=1)      # (N,)
basic_norm = np.linalg.norm(basic_np, axis=1)
after_norm = np.linalg.norm(after_np, axis=1)

# 4. 클러스터링 (예: K=128, 필요시 조정)
K = 128
base_kmeans = MiniBatchKMeans(n_clusters=K, batch_size=4096, random_state=42)
after_kmeans = MiniBatchKMeans(n_clusters=K, batch_size=4096, random_state=42)

base_cluster_id = base_kmeans.fit_predict(basic_np)   # (N,)
after_cluster_id = after_kmeans.fit_predict(after_np)

# 중심까지 거리 (군집 밀도/중심부 vs 바깥쪽 판별용)
base_center = base_kmeans.cluster_centers_[base_cluster_id]
after_center = after_kmeans.cluster_centers_[after_cluster_id]

base_dist2center = np.linalg.norm(basic_np - base_center, axis=1)
after_dist2center = np.linalg.norm(after_np - after_center, axis=1)

# 5. feature 테이블 구성
df_feat = df_label.copy()
df_feat["drift_norm"] = drift
df_feat["basic_norm"] = basic_norm
df_feat["after_norm"] = after_norm
df_feat["base_cluster_id"] = base_cluster_id
df_feat["after_cluster_id"] = after_cluster_id
df_feat["base_dist2center"] = base_dist2center
df_feat["after_dist2center"] = after_dist2center

# 저장
df_feat.to_csv("emb_features_basic_after.csv", index=False)