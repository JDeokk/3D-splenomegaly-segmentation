# postprocess.py
import os
import json
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.ndimage import label

def extract_id_and_month(file_identifier):
    """
    파일명에서 ID와 월 정보를 추출
    """
    parts = file_identifier.split('_')
    id_folder = "_".join(parts[:2])
    month_token = None
    for part in parts:
        if "개월" in part:
            month_token = part
            break
    if month_token is not None:
        month_folder = month_token.replace("개월", "")
    else:
        month_folder = "before"
    return id_folder, month_folder

def postprocess_and_visualize_segmentation(image_identifier, base_nifti_dir="data/nifti",
                                             segmentation_dir="outputs/test1", result_dir="outputs/result"):
    """
    원본 NIfTI와 세그멘테이션 결과를 불러와 후처리 수행
    부피 측정 및 각 슬라이스에 대해 원본과 마스크를 오버레이하여 시각화
    """
    original_image_path = os.path.join(base_nifti_dir, image_identifier.split('_')[0], f"{image_identifier}.nii.gz")
    segmentation_image_path = os.path.join(segmentation_dir, f"{image_identifier}.nii.gz")
    post_processed_image_path = os.path.join(result_dir, f"{image_identifier}_postprocessed.nii.gz")
    os.makedirs(os.path.dirname(post_processed_image_path), exist_ok=True)

    print("후처리를 위한 NIfTI 파일 로드 중...")
    original_img = nib.load(original_image_path)
    original_image = original_img.get_fdata()
    segmentation_img = nib.load(segmentation_image_path)
    segmentation_image = segmentation_img.get_fdata()
    print("파일 로드 완료.\n")

    label_of_interest = 1
    print("관심 라벨 필터링 진행...")
    filtered_segmentation = np.where(segmentation_image == label_of_interest, 1, 0)
    print("필터링 완료.\n")

    print("연결 성분 분석 수행 중...")
    labeled_seg, num_features = label(filtered_segmentation)
    print(f"발견된 연결 성분 수: {num_features}")
    sizes = np.bincount(labeled_seg.ravel())
    sizes[0] = 0
    largest_label = sizes.argmax()
    print(f"가장 큰 연결 성분: 라벨 {largest_label} (크기: {sizes[largest_label]} voxels)")
    largest_component = np.where(labeled_seg == largest_label, 1, 0)
    print("연결 성분 분석 완료.\n")

    filled_segmentation = scipy.ndimage.binary_fill_holes(largest_component).astype(np.uint8)

    print(f"후처리된 세그멘테이션을 {post_processed_image_path}에 저장 중...")
    post_processed_img = nib.Nifti1Image(filled_segmentation.astype(np.uint8),
                                          affine=segmentation_img.affine,
                                          header=segmentation_img.header)
    nib.save(post_processed_img, post_processed_image_path)
    print("후처리 파일 저장 완료.\n")

    print("장기 부피 측정 진행 중...")
    voxel_dimensions = segmentation_img.header.get_zooms()
    voxel_volume_mm3 = np.prod(voxel_dimensions)
    num_voxels = np.sum(filled_segmentation == 1)
    total_volume_mm3 = num_voxels * voxel_volume_mm3
    total_volume_cm3 = total_volume_mm3 / 1000
    total_volume_liters = total_volume_cm3 / 1000

    print(f"Voxel 크기 (mm): {voxel_dimensions}")
    print(f"Voxel 부피: {voxel_volume_mm3:.2f} mm³")
    print(f"관심 라벨의 voxel 수: {num_voxels}")
    print(f"총 부피: {total_volume_mm3:.2f} mm³ ({total_volume_cm3:.2f} cm³, {total_volume_liters:.4f} liters)\n")

    id_folder, month_folder = extract_id_and_month(image_identifier)
    legend_label = f"{id_folder}_{month_folder}month"
    save_dir = os.path.join("visualizations_1", id_folder, month_folder)
    os.makedirs(save_dir, exist_ok=True)
    num_slices = original_image.shape[2]

    for z in range(num_slices):
        original_slice = original_image[:, :, z]
        mask_slice = filled_segmentation[:, :, z]

        plt.figure(figsize=(6, 6))
        plt.imshow(original_slice.T, cmap='gray', origin='lower')
        plt.imshow(mask_slice.T, cmap='Reds', alpha=0.3)
        plt.title(f"{legend_label} - Slice {z}\nTotal Volume: {total_volume_cm3:.2f} cm³")
        plt.axis('off')
        save_path = os.path.join(save_dir, f"slice_{z}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    print(f"모든 슬라이스 시각화 완료. 저장 경로: {save_dir}\n")
    print(f"계산된 부피: {total_volume_cm3:.2f} cm³\n")
    print("후처리 및 시각화 완료.\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="후처리 및 시각화 실행")
    parser.add_argument("--image_identifier", type=str, default=None)
    parser.add_argument("--json_path", type=str, default=None)
    parser.add_argument("--image_index", type=int, default=None)
    parser.add_argument("--base_nifti_dir", default="data/nifti", type=str, help="원본 NIfTI 파일 디렉토리")
    parser.add_argument("--segmentation_dir", default="outputs/test1", type=str, help="세그멘테이션 결과 디렉토리")
    parser.add_argument("--result_dir", default="outputs/result", type=str, help="후처리 결과 저장 디렉토리")
    args = parser.parse_args()

    if args.json_path:
        with open(args.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        test_list = data.get("test", [])
        if args.image_index is not None:
            test_list = [test_list[args.image_index]]
        for test_file in test_list:
            base = os.path.basename(test_file)
            if base.endswith('.nii.gz'):
                image_identifier = base[:-7]
            elif base.endswith('.nii'):
                image_identifier = base[:-4]
            else:
                image_identifier = base
            postprocess_and_visualize_segmentation(image_identifier,
                                                   base_nifti_dir=args.base_nifti_dir,
                                                   segmentation_dir=args.segmentation_dir,
                                                   result_dir=args.result_dir)
    else:
        if args.image_identifier is None:
            raise ValueError("json_path 또는 image_identifier 중 하나를 제공해야 합니다.")
        postprocess_and_visualize_segmentation(args.image_identifier,
                                               base_nifti_dir=args.base_nifti_dir,
                                               segmentation_dir=args.segmentation_dir,
                                               result_dir=args.result_dir)
