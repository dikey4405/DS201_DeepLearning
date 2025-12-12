import json
import os
import sys
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from tqdm import tqdm

# --- CẤU HÌNH ---
# 1. File Ground Truth gốc (để lấy caption chuẩn)
SOURCE_TEST_JSON = '/kaggle/input/data-dl/Captions/test.json' 

# 2. File kết quả sinh ra từ generate.py
RESULT_JSON_INPUT = 'caption_readable.json' 
if not os.path.exists(RESULT_JSON_INPUT):
    # Nếu không thấy file readable, dùng file kết quả chuẩn
    RESULT_JSON_INPUT = '/kaggle/working/caption_results.json'

GROUND_TRUTH_COCO_JSON = 'coco_vietnamese_gt_test.json'
RESULT_COCO_JSON = 'coco_results_formatted.json'

def _reformat_annotations(raw_annotations):
    """Chuyển đổi list annotation gốc thành dict {image_name: [list_captions]}"""
    reformatted = {}
    for item in raw_annotations:
        img_name = item.get('image_name')
        viet_cap = item.get('translate') # Key chứa caption tiếng Việt
        
        if img_name and viet_cap:
            if img_name not in reformatted:
                reformatted[img_name] = []
            reformatted[img_name].append(viet_cap.lower().strip())
    return reformatted

def convert_gt_to_coco(source_json_path, output_json_path):
    """Tạo file Ground Truth theo chuẩn COCO (cần ID số nguyên)."""
    print(f"Đang tạo file Ground Truth chuẩn COCO từ: {source_json_path}")
    
    with open(source_json_path, 'r', encoding='utf-8') as f:
        raw_annotations = json.load(f)
    
    data_dict = _reformat_annotations(raw_annotations)
    
    images = []
    annotations = []
    
    # Tạo mapping: filename (string) -> id (int)
    # Vì COCO yêu cầu image_id phải là số nguyên (hoặc string số)
    filename_to_id = {}
    
    ann_id = 1
    for idx, (filename, caps) in enumerate(tqdm(data_dict.items(), desc="Formatting GT")):
        # Tạo ID ảo dựa trên index (1, 2, 3...)
        img_id = idx + 1
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
    """Chuyển file dự đoán sang chuẩn COCO Results (khớp ID với file GT)."""
    print(f"Đang định dạng file dự đoán: {pred_json_path}")
    
    with open(pred_json_path, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    
    coco_results = []
    
    # Xử lý input tùy theo format (Dict hay List)
    if isinstance(preds, dict):
        # Format: {"filename.jpg": "caption"}
        iterator = preds.items()
    elif isinstance(preds, list):
        # Format: [{"image_id": "filename.jpg", "caption": "..."}]
        iterator = [(p['image_id'], p['caption']) for p in preds]
    else:
        print("Lỗi: Định dạng file kết quả không hợp lệ.")
        return False

    for filename, caption in iterator:
        # Tìm ID số nguyên tương ứng của filename này
        img_id = filename_to_id.get(filename)
        
        if img_id:
            coco_results.append({
                'image_id': img_id,
                'caption': str(caption).lower().strip()
            })
            
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_results, f, ensure_ascii=False)
        
    return True

def evaluate(gt_path, res_path):
    # Suppress print của thư viện COCO (nếu muốn gọn)
    # sys.stdout = open(os.devnull, 'w') 
    
    try:
        # 1. Load GT
        coco = COCO(gt_path)
        # 2. Load Res
        coco_res = coco.loadRes(res_path)
        
        # 3. Eval
        coco_eval = COCOEvalCap(coco, coco_res)
        coco_eval.params['image_id'] = coco_res.getImgIds()
        coco_eval.evaluate()
        
        # In kết quả đẹp
        print("\n" + "="*40)
        print("   KẾT QUẢ ĐÁNH GIÁ (SCST + Faster R-CNN)")
        print("="*40)
        for metric, score in coco_eval.eval.items():
            print(f"| {metric:10s} | {score*100:6.2f}% |")
        print("="*40)
        
    except Exception as e:
        print(f"\n[LỖI EVAL]: {e}")
        print("Gợi ý: Nếu lỗi SPICE, hãy kiểm tra Java đã cài chưa.")
    # finally:
        # sys.stdout = sys.__stdout__ # Restore print

if __name__ == '__main__':
    # 1. Tạo file GT chuẩn COCO (và lấy mapping ID)
    if os.path.exists(SOURCE_TEST_JSON):
        filename_to_id = convert_gt_to_coco(SOURCE_TEST_JSON, GROUND_TRUTH_COCO_JSON)
    else:
        print(f"Lỗi: Không tìm thấy file gốc {SOURCE_TEST_JSON}"); exit()

    # 2. Chuẩn hóa file dự đoán
    if os.path.exists(RESULT_JSON_INPUT):
        success = convert_pred_to_coco(RESULT_JSON_INPUT, RESULT_COCO_JSON, filename_to_id)
        if not success: exit()
    else:
        print(f"Lỗi: Không tìm thấy file kết quả {RESULT_JSON_INPUT}. Chạy generate.py chưa?"); exit()

    # 3. Chạy đánh giá
    evaluate(GROUND_TRUTH_COCO_JSON, RESULT_COCO_JSON)
