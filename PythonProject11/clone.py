import numpy as np


def create_projection_matrix(fx=1000, fy=1000, cx=320, cy=240,
                             shift_x=0, shift_y=0, shift_z=0):
    """Tạo ma trận chiếu (3x4) cho camera với các tham số đầu vào.

    Args:
        fx, fy: Tiêu cự theo trục x và y
        cx, cy: Tâm ảnh theo trục x và y
        shift_x, shift_y, shift_z: Vector tịnh tiến camera
    """
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    # Sử dụng np.column_stack thay cho np.hstack cho rõ ràng hơn
    Rt = np.column_stack((np.eye(3), [[shift_x], [shift_y], [shift_z]]))
    return K @ Rt


def main():
    # Định nghĩa các camera với tham số rõ ràng
    cameras = {
        'top': {'shift_x': 0, 'shift_y': 0, 'shift_z': 0},
        'left': {'shift_x': -200, 'shift_y': 0, 'shift_z': 0},
        'right': {'shift_x': 200, 'shift_y': 0, 'shift_z': 0}
    }

    # Tạo và lưu từng ma trận
    for name, params in cameras.items():
        P = create_projection_matrix(**params)
        np.save(f'P_{name}.npy', P)
        print(f"Đã tạo và lưu P_{name}.npy")


if __name__ == '__main__':
    main()