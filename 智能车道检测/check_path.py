import os

test_images_dir = r"E:\PythonProject\计算机视觉\智能车道检测\test_images"
test_videos_dir = r"E:\PythonProject\计算机视觉\智能车道检测\test_videos"

print(f"测试图片目录: {test_images_dir}")
print(f"目录存在: {os.path.exists(test_images_dir)}")

if os.path.exists(test_images_dir):
    files = os.listdir(test_images_dir)
    print(f"目录中的文件: {files}")
else:
    print("测试图片目录不存在")

print(f"\n测试视频目录: {test_videos_dir}")
print(f"目录存在: {os.path.exists(test_videos_dir)}")

if os.path.exists(test_videos_dir):
    files = os.listdir(test_videos_dir)
    print(f"目录中的文件: {files}")
else:
    print("测试视频目录不存在")