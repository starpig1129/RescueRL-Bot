import cv2
import numpy as np
from DataReader import DataReader

class DetectionResult:
    def __init__(self, epoch, frame_idx, frame, boxes, scores, classes):
        self.epoch = epoch
        self.frame_idx = frame_idx
        self.frame = frame
        self.boxes = boxes
        self.scores = scores
        self.classes = classes

def display_detection(frame, boxes, scores, classes):
    """顯示檢測結果"""
    if boxes is None or len(boxes) == 0:
        return frame

    frame_with_box = frame.copy()
    for box, score, cls in zip(boxes, scores, classes):
        if score > 0:  # 只顯示有效的檢測
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_with_box, f"{score:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame_with_box

def find_detections(start_epoch, end_epoch, confidence_threshold=0.5):
    """找出指定世代範圍內所有成功偵測到人的畫面"""
    print(f"開始搜尋世代 {start_epoch} 到 {end_epoch} 的偵測結果...")
    data_reader = DataReader()
    detection_results = []
    
    for epoch in range(start_epoch, end_epoch + 1, 50):  # 每50個世代為一個間隔
        print(f"\n處理世代 {epoch}...")
        
        # 獲取該世代的數據
        max_steps = data_reader.get_max_steps(epoch)
        if max_steps is None:
            print(f"無法讀取世代 {epoch} 的數據，跳過")
            continue
        
        print(f"讀取世代 {epoch} 的數據 (共 {max_steps} 步)...")
        data = data_reader.load_range_data(epoch, slice(0, max_steps))
        if data is None:
            print(f"無法載入世代 {epoch} 的數據，跳過")
            continue
        
        # 遍歷所有幀，尋找成功的偵測結果
        total_frames = len(data['obs'])
        detection_count = 0
        
        for frame_idx in range(total_frames):
            scores = data['yolo_scores'][frame_idx]
            if scores is not None and len(scores) > 0 and np.max(scores) > confidence_threshold:
                detection_count += 1
                detection_results.append(DetectionResult(
                    epoch=epoch,
                    frame_idx=frame_idx,
                    frame=data['origin_image'][frame_idx],
                    boxes=data['yolo_boxes'][frame_idx],
                    scores=scores,
                    classes=data['yolo_classes'][frame_idx]
                ))
            
            if frame_idx % 1000 == 0:
                print(f"進度: {frame_idx}/{total_frames} | 已找到 {detection_count} 個檢測結果")
        
        print(f"世代 {epoch} 完成，找到 {detection_count} 個檢測結果")
    
    return detection_results

def play_detections(detection_results):
    """播放所有檢測結果"""
    if not detection_results:
        print("沒有找到任何檢測結果")
        return
    
    print(f"\n開始播放 {len(detection_results)} 個檢測結果...")
    print("按 'q' 退出，空白鍵暫停/繼續，左右方向鍵調整播放速度")
    
    cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detections', 1280, 720)
    
    paused = False
    current_idx = 0
    frame_delay = 100  # 初始延遲100ms
    
    while current_idx < len(detection_results):
        if not paused:
            result = detection_results[current_idx]
            
            # 顯示偵測結果
            frame_with_detection = display_detection(
                result.frame, result.boxes, result.scores, result.classes)
            
            # 添加資訊 (更簡潔的格式)
            info_text = f"E{result.epoch} F{result.frame_idx} [{current_idx + 1}/{len(detection_results)}]"
            cv2.putText(frame_with_detection, info_text, 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Detections', frame_with_detection)
            current_idx += 1
        
        # 等待按鍵輸入
        key = cv2.waitKey(frame_delay)
        
        if key == ord('q'):  # 按 q 退出
            break
        elif key == ord(' '):  # 按空白鍵暫停/繼續
            paused = not paused
        elif key == 81:  # 左方向鍵，降低播放速度
            frame_delay = min(500, frame_delay + 50)
            print(f"播放延遲: {frame_delay}ms")
        elif key == 83:  # 右方向鍵，提高播放速度
            frame_delay = max(1, frame_delay - 50)
            print(f"播放延遲: {frame_delay}ms")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='顯示成功偵測到人的畫面')
    parser.add_argument('--start-epoch', type=int, required=True, help='起始世代')
    parser.add_argument('--end-epoch', type=int, required=True, help='結束世代')
    parser.add_argument('--threshold', type=float, default=0.5, help='偵測信心度閾值')
    
    args = parser.parse_args()
    
    # 先找出所有檢測結果
    results = find_detections(args.start_epoch, args.end_epoch, args.threshold)
    print(f"\n共找到 {len(results)} 個檢測結果")
    
    # 播放所有結果
    if results:
        play_detections(results)
