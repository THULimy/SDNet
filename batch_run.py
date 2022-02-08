from joblib import Parallel, delayed
import queue
import os

os.environ['MKL_THREADING_LAYER'] = 'GNU'


# Define number of GPUs available
N_GPU = [0, 1, 2, 3, 4]

# data = 'cifar'  # cifar, cifar100, tiny_imagenet
# log_title = f'{data}_sdnet18'


# CUDA_VISIBLE_DEVICES=2 python train.py --cfg experiments/debug.yaml --dir_phase debug_cifar_lenet8_niteration2_adaptive_lmbd0.5
# MODEL.NUM_LAYERS 2 TRAIN.LR 0.1 MODEL.NAME lenet8_viz MODEL.LAMBDA 0.5,

# Put here your job
def build_cmd():
    # train
    base = f'python train.py --cfg experiments/debug.yaml'
    cmd = []
    # factors
    for var1 in [16]:  # num-iteration in ista algorithm
        for var2 in [0.0]:  # lambda
            for var3 in [False]:  # adaptive
                for var4 in [0.1]:
                    dir_phase = f' --dir_phase debug_cifar_lenet8_niteration{var1}_lr{var4}_lmbd{var2}'
                    cmd_info = f' MODEL.NUM_LAYERS {var1} MODEL.NAME lenet8_viz MODEL.LAMBDA {var2}, MODEL.POOLING False TRAIN.LR {var4} '
                    cmd.append(base + dir_phase + cmd_info)

    # visualize
    base = f'python visualize.py --cfg experiments/debug.yaml'  # sdnet.yaml
    cmd = []
    # factors
    for var1 in [4]:  # num-iteration in ista algorithm  4, 8, 16, 32, 64
        for var2 in [0.0001, 0.001, 0.01, 0.1]:  # lambda  0.00001, 0.0001, 0.001, 0.01, 0.1
            for var3 in [False]:  # adaptive
                for var4 in [0.1]:
                    name = f'debug_cifar_lenet8_niteration{var1}_lr{var4}_lmbd{var2}'  # imagenet_sdnet18_viz_niteration2
                    log_phase = f' --dir_phase viz4{name}'
                    ckpt = f' TRAIN.MODEL_FILE logs/{name}/model_best.pth.tar'
                    cmd_info = f' MODEL.NUM_LAYERS {var1} MODEL.NAME lenet8_viz MODEL.LAMBDA {var2}, MODEL.POOLING False TRAIN.LR {var4}'
                    cmd.append(base + log_phase + cmd_info + ckpt)

    # base = f'python visualize.py --cfg experiments/sdnet_imagenet.yaml'  # sdnet.yaml  sdnet_imagenet.yaml
    # cmd = []
    # # factors
    # for var1 in [2]:  # num-iteration in ista algorithm
    #     for var2 in [True]:  # train set
    #         for var3 in [True]:  # input norm
    #             name = 'debug_cifar_lenet_niteration10'  # imagenet_sdnet18_viz_niteration2
    #             log_phase = f' --dir_phase viz4{name}_trainset{var2}_inputnorm{var3}'
    #             ckpt = f' TRAIN.MODEL_FILE logs/{name}/model_best.pth.tar'
    #             cmd_info = f' MODEL.SHORTCUT False MODEL.NUM_LAYERS {var1} VIZ_TRAINSET {var2} VIZ_INPUTNORM {var3}' + ' TRAIN.BATCH_SIZE_PER_GPU 10'
    #             cmd.append(base + log_phase + cmd_info + ckpt)

    return cmd

cmd = build_cmd()


# Put indices in queue
q = queue.Queue(maxsize=len(N_GPU))
for i in N_GPU:
    q.put(i)


def runner(x):
    gpu = q.get()

    # current_cmd = cmd[-(x + 1)]
    current_cmd = cmd[x]
    print(gpu, current_cmd)
    os.system("CUDA_VISIBLE_DEVICES={} {}".format(gpu, current_cmd))

    # return gpu id to queue
    q.put(gpu)


def collect_logs():
    prefix = "cifar_sdnet18_niteration8_"
    regramma = f"logs/{prefix}*"
    out_pth = f"logs/{prefix}"

    import glob
    pths = glob.glob(regramma)

    os.makedirs(out_pth, exist_ok=True)
    for p in pths:
        dir_name = os.path.basename(p)
        os.system(f"mkdir {out_pth}/{dir_name}")
        os.system(f"cp {p}/*.log {out_pth}/{dir_name}/")


if __name__ == '__main__':
    # collect_logs()
    # exit()

    # Change loop
    Parallel(n_jobs=len(N_GPU), backend="threading")(
        delayed(runner)(i) for i in range(len(cmd)))

    # collect_logs()
