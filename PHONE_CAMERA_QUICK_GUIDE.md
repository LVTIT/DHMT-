# Hướng Dẫn Nhanh: Điện Thoại Làm Camera

Dành cho demo Human Motion Capture.

## 1. Chạy Dự Án
```powershell
cd d:\project
.\.venv\Scripts\Activate.ps1
uvicorn main:app --reload
```
Mở trình duyệt: `http://localhost:8000`

Lỗi kẹt port 8000:
```powershell
Get-NetTCPConnection -LocalPort 8000 | Select-Object OwningProcess
Stop-Process -Id [PID] -Force
uvicorn main:app --reload
```

## 2. iPhone (Camo)
1. Cài Camo (iPhone) & Camo Studio (Windows).
2. Cắm cáp USB/lighning , chọn `Trust This Computer`.
3. Web: nhập source `1@MSMF` -> áp dụng.
4. Mode: `9:16`, `contain`, `0 deg`.
*Mẹo:* Chọn 720p/30FPS cho mượt. Nếu màn đen, bấm scan chọn MSMF khác.

## 3. Android (DroidCam)
1. Cài DroidCam cho Android & Windows, rồi kết nối.
2. Web: bấm scan, thử `1@MSMF`, `2@MSMF`... tới khi có hình.
*Mode:* ĐT dựng dọc -> `9:16`, ĐT nằm ngang -> `16:9` (kèm `contain` & `0 deg`).

## 4. Android (Dùng App URL - IP Camera)
1. Kết nối chung mạng Wi-Fi, copy URL app (VD: `http://192.168.1.10:8080/video`).
2. Dán URL vào ô source -> áp dụng.

## 5. Chọn Mode Detect
- Luôn ưu tiên chế độ `contain` thay vì `cover` để lấy trọn người.
- Đứng lùi 1.5 - 2.5m tránh mất chân tay, đảm bảo đủ sáng.

## 6. Fix lỗi nhanh
- **Camera đen:** Đóng Zoom, Teams. Mở app ĐT và phần mềm PC trước mới chọn source ở Web.
- **Delay:** Chỉnh phân giải app về 720p/30FPS. Cáp USB luôn nhanh hơn Wi-Fi.
- **Tắt server:** `Ctrl + C`.
- **Reset log tay:**
```powershell
Set-Content logs\gesture_log.csv "timestamp,detected_pose,confidence"
```
