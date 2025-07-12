import subprocess
import sys

# Danh sách thư viện và phiên bản cần cài đặt
packages = {
    "numpy": "1.26.4",
    "opencv-python": "4.11.0.86",
    "mediapipe": "0.10.21",
    "customtkinter": "5.2.2",
    "paramiko": "3.5.1",
    "requests": "2.32.4",
    "pandas": "2.3.0",
    "scikit-learn": "1.6.1",
    "tensorflow": "2.19.0",
    "torch": "2.7.1+cu118",
    "torchaudio": "2.7.1+cu118",
    "torchvision": "0.22.1+cu118",
    "ultralytics": "8.3.162"
}

def install(package, version):
    package_spec = f"{package}=={version}"
    print(f"Đang cài {package_spec} ...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
        print(f"✔️  Cài {package_spec} thành công!")
    except subprocess.CalledProcessError:
        print(f"❌ Cài {package_spec} thất bại!")

if __name__ == "__main__":
    print("Bắt đầu cài đặt đúng phiên bản các thư viện cần thiết...\n")
    for package, version in packages.items():
        install(package, version)
    print("\n🎉 Đã hoàn tất cài đặt toàn bộ thư viện theo đúng phiên bản!")
