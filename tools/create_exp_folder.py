import os

def create_exp_folder():
    # Step 1: 创建run文件夹（如果不存在）
    if not os.path.exists("run"):
        os.mkdir("run")

    # Step 2: 创建train文件夹（如果不存在）
    train_folder = os.path.join("run", "train")
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)

    # Step 3: 创建exp文件夹（检查是否存在）
    exp_folder = os.path.join(train_folder, "exp")
    if not os.path.exists(exp_folder):
        os.mkdir(exp_folder)
        os.mkdir(os.path.join(exp_folder, "weights"))  # 创建weights文件夹
        return exp_folder, os.path.join(exp_folder, "weights")  # 返回exp和weights文件夹路径

    # 如果exp文件夹已存在，则查找exp1, exp2, 等
    exp_num = 1
    while True:
        # 动态命名exp1, exp2, ...
        exp_folder_name = f"exp{exp_num}"
        exp_folder = os.path.join(train_folder, exp_folder_name)
        if not os.path.exists(exp_folder):
            os.mkdir(exp_folder)  # 创建新的exp文件夹
            os.mkdir(os.path.join(exp_folder, "weights"))  # 创建weights文件夹
            return exp_folder, os.path.join(exp_folder, "weights")  # 返回exp和weights文件夹路径
        exp_num += 1  # 如果文件夹已存在，增加数字，继续查找下一个文件夹


def create_val_exp_folder():
    run_dir = "run"
    os.makedirs(run_dir, exist_ok=True)

    predict_dir = os.path.join(run_dir, "predict")
    os.makedirs(predict_dir, exist_ok=True)

    # 先用 exp
    exp0 = os.path.join(predict_dir, "exp")
    if not os.path.exists(exp0):
        os.makedirs(exp0)
        return exp0

    # 再用 exp1, exp2 ...
    exp_num = 1
    while True:
        exp_folder = os.path.join(predict_dir, f"exp{exp_num}")
        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)
            return exp_folder
        exp_num += 1


