import json
import os
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from tqdm import tqdm

# --- CẤU HÌNH ---
# File JSON nguồn của tập TEST
SOURCE_TEST_JSON = './data/test_captions.json' 
# File JSON Ground Truth COCO chuẩn (sẽ được tạo)
GROUND_TRUTH_JSON = './data/coco_vietnamese_gt_test.json'
# File JSON Results (được tạo bởi generate.py)
RESULT_JSON = './caption_results.json' 

def _reformat_annotations(raw_annotations):
    """Tái cấu trúc file JSON nguồn (dạng List) thành dictionary {image_name: [viet_cap, ...]}"""
    reformatted = {}
    for item in raw_annotations:
        img_name = item.get('image_name')
        viet_cap = item.get('translate') 
        if img_name and viet_cap:
            if img_name not in reformatted:
                reformatted[img_name] = []
            reformatted[img_name].append(viet_cap.lower().strip())
    return reformatted

def convert_annotations_to_coco_format(source_json_path, output_json_path):
    """Chuyển đổi file JSON nguồn của bạn thành định dạng JSON COCO Annotations chuẩn."""
    print(f"Bắt đầu chuyển đổi JSON nguồn ({source_json_path}) thành COCO GT...")
    
    with open(source_json_path, 'r', encoding='utf-8') as f:
        raw_annotations = json.load(f)
    reformatted_data = _reformat_annotations(raw_annotations)
    
    annotations = []
    images = []
    annotation_id = 1
    image_names = list(reformatted_data.keys())
    name_to_id = {name: i + 1 for i, name in enumerate(image_names)}
    
    for name in tqdm(image_names, desc="Tạo cấu trúc COCO"):
        img_id = name_to_id[name]
        images.append({'id': img_id, 'file_name': name})
        caps = reformatted_data[name]
        for caption in caps:
            annotations.append({'id': annotation_id, 'image_id': img_id, 'caption': caption})
            annotation_id += 1

    coco_format = {'info': {}, 'licenses': [], 'images': images, 'annotations': annotations}
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_format, f, ensure_ascii=False, indent=4)
    print(f"Hoàn thành. JSON Ground Truth đã lưu tại: {output_json_path}")
    return name_to_id

def create_result_json(generated_captions_dict, name_to_id):
    """Chuyển đổi dictionary kết quả sinh ra {image_name: caption} sang định dạng JSON COCO Results."""
    print("Tạo JSON Results...")
    coco_results = []
    for img_name, caption in generated_captions_dict.items():
        if img_name in name_to_id:
            img_id = name_to_id[img_name]
            coco_results.append({'image_id': img_id, 'caption': str(caption).lower().strip()})
    
    with open(RESULT_JSON, 'w', encoding='utf-8') as f:
        json.dump(coco_results, f, ensure_ascii=False, indent=4)
    print(f"JSON Results đã lưu tại: {RESULT_JSON}")
    return RESULT_JSON

def evaluate_captions(ground_truth_json, result_json):
    """Thực hiện đánh giá bằng COCOEvalCap."""
    print("\n--- BẮT ĐẦU ĐÁNH GIÁ ---")
    try:
        coco = COCO(ground_truth_json)
        coco_res = coco.loadRes(result_json)
        coco_eval = COCOEvalCap(coco, coco_res)
        coco_eval.params['image_id'] = coco_res.getImgIds()
        coco_eval.evaluate()
        
        print("\n--- KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG ---")
        metrics = {}
        for metric, score in coco_eval.eval.items():
            print(f"{metric}: {score*100:.2f}%")
            metrics[metric] = score
        return metrics
    except Exception as e:
        print(f"Lỗi trong quá trình đánh giá COCOEvalCap: {e}")
        return None

if __name__ == '__main__':
    # 1. Chuẩn bị Ground Truth (Chỉ chạy 1 lần)
    if not os.path.exists(GROUND_TRUTH_JSON):
        name_to_id_map = convert_annotations_to_coco_format(SOURCE_TEST_JSON, GROUND_TRUTH_JSON)
    else:
        # Tải lại name_to_id map nếu file GT đã tồn tại
        with open(GROUND_TRUTH_JSON, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
            name_to_id_map = {img['file_name']: img['id'] for img in gt_data['images']}
        print("Đã tìm thấy file Ground Truth JSON.")

    # 2. Tạo JSON Results (từ file đã sinh)
    try:
        with open(RESULT_JSON, 'r', encoding='utf-8') as f:
            generated_results_dict = json.load(f)
        create_result_json(generated_results_dict, name_to_id_map)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {RESULT_JSON}. Vui lòng chạy generate.py trước.")
        exit()
        
    # 3. Đánh giá
    evaluate_captions(GROUND_TRUTH_JSON, RESULT_JSON)