import json
import os
import sys
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from tqdm import tqdm

# --- CẤU HÌNH ---
SOURCE_TEST_JSON = '/kaggle/input/data-dl/Captions/test.json' 
RESULT_JSON_INPUT = 'caption_readable.json' 

if not os.path.exists(RESULT_JSON_INPUT):
    RESULT_JSON_INPUT = '/kaggle/working/caption_results.json'

GROUND_TRUTH_COCO_JSON = 'coco_vietnamese_gt_test.json'
RESULT_COCO_JSON = 'coco_results_formatted.json'

def _reformat_annotations(raw_annotations):
    """Chuyển đổi list annotation gốc thành dict {image_name: [list_captions]}"""
    reformatted = {}
    for item in raw_annotations:
        img_name = item.get('image_name')
        viet_cap = item.get('translate')
        
        if img_name and viet_cap:
            if img_name not in reformatted:
                reformatted[img_name] = []
            
            # === [QUAN TRỌNG 1] Xử lý Ground Truth ===
            # Thêm .replace("_", " ") để tách từ ghép (vd: con_mèo -> con mèo)
            # Điều này giúp khớp với tokenizer của COCOEval
            clean_cap = viet_cap.replace("_", " ").lower().strip()
            reformatted[img_name].append(clean_cap)
            
    return reformatted

def convert_gt_to_coco(source_json_path, output_json_path):
    """Tạo file Ground Truth chuẩn COCO."""
    print(f"Đang tạo file Ground Truth chuẩn COCO từ: {source_json_path}")
    
    with open(source_json_path, 'r', encoding='utf-8') as f:
        raw_annotations = json.load(f)
    
    data_dict = _reformat_annotations(raw_annotations)
    
    images = []
    annotations = []
    filename_to_id = {}
    
    ann_id = 1
    for idx, (filename, caps) in enumerate(tqdm(data_dict.items(), desc="Formatting GT")):
        img_id = idx + 1 # Tạo ID số nguyên
        filename_to_id[filename] = img_id
        
        images.append({'id': img_id, 'file_name': filename})
        
        for c in caps:
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'caption': c
            })
            ann_id += 1
            
    coco_format = {
        'info': {'description': 'Vietnamese Captions GT'},
        'licenses': [],
        'images': images,
        'annotations': annotations
    }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_format, f, ensure_ascii=False)
        
    return filename_to_id

def convert_pred_to_coco(pred_json_path, output_json_path, filename_to_id):
    """Chuyển file dự đoán sang chuẩn COCO Results."""
    print(f"Đang định dạng file dự đoán: {pred_json_path}")
    
    with open(pred_json_path, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    
    coco_results = []
    
    if isinstance(preds, dict):
        iterator = preds.items()
    elif isinstance(preds, list):
        iterator = [(p['image_id'], p['caption']) for p in preds]
    else:
        print("Lỗi: Định dạng file kết quả không hợp lệ.")
        return False

    for filename, caption in iterator:
        # Nếu filename trong result là số (vd: "123") thì convert sang int để khớp
        # Nếu là tên file (vd: "COCO_val_...jpg") thì tra từ điển
        img_id = filename_to_id.get(filename)
        
        # Fallback: nếu filename_to_id không tìm thấy, thử ép kiểu int trực tiếp
        # (Phòng trường hợp generate.py lưu ID là số string "123" thay vì tên file)
        if img_id is None:
             try:
                 if isinstance(filename, int) or filename.isdigit():
                     # Kiểm tra xem ID này có trong tập GT không (để tránh lỗi)
                     possible_id = int(filename)
                     if possible_id in filename_to_id.values():
                         img_id = possible_id
             except:
                 pass

        if img_id:
            # === [QUAN TRỌNG 2] Xử lý Prediction ===
            # Cũng phải thay thế _ bằng khoảng trắng để đồng bộ với Ground Truth
            clean_pred = str(caption).replace("_", " ").lower().strip()
            
            coco_results.append({
                'image_id': img_id,
                'caption': clean_pred
            })
            
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_results, f, ensure_ascii=False)
        
    return True

def evaluate(gt_path, res_path):
    try:
        # 1. Load GT
        coco = COCO(gt_path)
        # 2. Load Res
        coco_res = coco.loadRes(res_path)
        
        # 3. Eval
        coco_eval = COCOEvalCap(coco, coco_res)
        coco_eval.params['image_id'] = coco_res.getImgIds()
        coco_eval.evaluate()
        
        print("\n" + "="*40)
        print("   KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG")
        print("="*40)
        for metric, score in coco_eval.eval.items():
            print(f"| {metric:10s} | {score*100:6.2f}% |")
        print("="*40)
        
    except Exception as e:
        print(f"\n[LỖI EVAL]: {e}")

if __name__ == '__main__':
    if os.path.exists(SOURCE_TEST_JSON):
        filename_to_id = convert_gt_to_coco(SOURCE_TEST_JSON, GROUND_TRUTH_COCO_JSON)
    else:
        print(f"Lỗi: Không tìm thấy file gốc {SOURCE_TEST_JSON}"); exit()

    if os.path.exists(RESULT_JSON_INPUT):
        success = convert_pred_to_coco(RESULT_JSON_INPUT, RESULT_COCO_JSON, filename_to_id)
        if not success: exit()
    else:
        print(f"Lỗi: Không tìm thấy file kết quả {RESULT_JSON_INPUT}"); exit()

    evaluate(GROUND_TRUTH_COCO_JSON, RESULT_COCO_JSON)
