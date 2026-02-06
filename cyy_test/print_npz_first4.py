import numpy as np


def to_python_list(arr, max_dim=4):
    """将 numpy 数组转换为 Python 列表格式"""
    if arr.ndim == 0:
        return arr.item()
    if arr.ndim == 1:
        return arr.tolist()
    # 递归处理多维数组
    return to_python_list(arr, max_dim - 1) if max_dim > 1 else arr.tolist()


def print_array(arr, max_dim=4, indent=0):
    """以 Python 数组风格打印数组"""
    if isinstance(arr, np.ndarray):
        if arr.ndim == 0:
            print(str(arr.item()), end='')
        elif arr.ndim == 1:
            print('[', end='')
            for i, val in enumerate(arr):
                if i < min(4, len(arr)):
                    print_array(val, max_dim - 1, 0)
                    if i < min(4, len(arr)) - 1:
                        print(', ', end='')
            if len(arr) > 4:
                print(', ...]', end='')
            else:
                print(']', end='')
        else:
            print('[', end='')
            for i in range(min(4, len(arr))):
                print_array(arr[i], max_dim - 1, 0)
                if i < min(4, len(arr)) - 1:
                    print(', ', end='')
            if len(arr) > 4:
                print(', ...]', end='')
            else:
                print(']', end='')
    else:
        # 标量 - 保留小数点后8位
        if isinstance(arr, (np.integer, np.floating)):
            val = float(arr)
            # 整数不显示小数点
            if val == int(val):
                print(int(val), end='')
            else:
                # 保留小数点后8位
                print(f"{val:.8f}", end='')
        else:
            print(arr, end='')


def print_npz(file_path, max_dim=4):
    """读取并打印 npz 文件内容"""
    data = np.load(file_path)

    print(f"文件: {file_path}")
    print("=" * 50)

    for key in data.keys():
        arr = data[key]
        print(f"\n[{key}]")
        print(f"  shape: {arr.shape}, dtype: {arr.dtype}")
        print(f"  值: ", end='')
        print_array(arr, max_dim)
        print()


if __name__ == "__main__":
    # 手动指定 npz 文件路径
    npz_file = "cyy_test/outputs/precision_test/vae_decoder_input_z_full.npz"

    # 可选: 设置最大输出维度，默认 4
    max_output_dim = 4

    print_npz(npz_file, max_output_dim)
