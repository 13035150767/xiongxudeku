import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
test_images_dir = os.path.join(SCRIPT_DIR, "test_images")
test_videos_dir = os.path.join(SCRIPT_DIR, "test_videos")
result_dir = os.path.join(SCRIPT_DIR, "结果")

print("=" * 60)
print("智能车道检测系统 - 路径检查")
print("=" * 60)

print(f"\n脚本目录: {SCRIPT_DIR}")
print(f"目录存在: {os.path.exists(SCRIPT_DIR)}")

print(f"\n测试图片目录: {test_images_dir}")
print(f"目录存在: {os.path.exists(test_images_dir)}")

if os.path.exists(test_images_dir):
    files = os.listdir(test_images_dir)
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_files = [f for f in files if f.lower().endswith(image_exts)]
    print(f"图片文件数量: {len(image_files)}")
    for f in image_files:
        file_path = os.path.join(test_images_dir, f)
        size_kb = os.path.getsize(file_path) / 1024
        print(f"  - {f} ({size_kb:.1f} KB)")
else:
    print("测试图片目录不存在")

print(f"\n测试视频目录: {test_videos_dir}")
print(f"目录存在: {os.path.exists(test_videos_dir)}")

if os.path.exists(test_videos_dir):
    files = os.listdir(test_videos_dir)
    video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    video_files = [f for f in files if f.lower().endswith(video_exts)]
    print(f"视频文件数量: {len(video_files)}")
    for f in video_files:
        file_path = os.path.join(test_videos_dir, f)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  - {f} ({size_mb:.1f} MB)")
else:
    print("测试视频目录不存在")

print(f"\n结果保存目录: {result_dir}")
print(f"目录存在: {os.path.exists(result_dir)}")

if os.path.exists(result_dir):
    files = os.listdir(result_dir)
    print(f"结果文件数量: {len(files)}")
    for f in files:
        file_path = os.path.join(result_dir, f)
        size_kb = os.path.getsize(file_path) / 1024
        print(f"  - {f} ({size_kb:.1f} KB)")
else:
    print("结果保存目录不存在（将在首次运行时自动创建）")

print("\n" + "=" * 60)
print("路径检查完成")
print("=" * 60)
