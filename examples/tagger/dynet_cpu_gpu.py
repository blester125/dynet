import multiprocessing as mp
import numpy as np

SEQ_LEN = 10
VSZ = 5
DSZ = 8

FILT = 3
OUT = 10

np.random.seed(1)

EMBED_WEIGHT = np.random.rand(VSZ, DSZ)
DYNET_EMBED = np.reshape(EMBED_WEIGHT, (VSZ, 1, 1, DSZ))

INPUT = np.random.randint(0, VSZ, size=SEQ_LEN)

CONV_WEIGHT = np.random.rand(1, FILT, DSZ, OUT)
CONV_BIAS = np.random.rand(OUT)

def gpu(return_dict):
    import _dynet
    dy_params = _dynet.DynetParams()
    dy_params.from_args()
    dy_params.set_requested_gpus(1)
    dy_params.set_random_seed(542380255)
    dy_params.init()
    return_dict['gpu_start'], return_dict['gpu'], return_dict['gpu_loss'] = run()

def cpu(return_dict):
    import _dynet
    dy_params = _dynet.DynetParams()
    dy_params.from_args()
    dy_params.set_random_seed(542380255)
    dy_params.init()
    return_dict['cpu_start'], return_dict['cpu'], return_dict['cpu_loss'] = run()

def run():
    import dynet as dy
    pc = dy.ParameterCollection()
    embed = pc.lookup_parameters_from_numpy(DYNET_EMBED)
    conv = pc.add_parameters((1, FILT, DSZ, OUT), init=dy.NumpyInitializer(CONV_WEIGHT))
    bias = pc.add_parameters(OUT, init=dy.NumpyInitializer(CONV_BIAS))
    trainer = dy.MomentumSGDTrainer(pc, 0.01, 0.9)
    trainer.set_sparse_updates(False)

    starts = [embed.npvalue(), INPUT, conv.npvalue(), bias.npvalue()]

    for _ in range(10000):
        dy.renew_cg()
        dy_embedded = [embed[x] for x in INPUT]
        dy_embedded = dy.concatenate(dy_embedded, d=1)

        dy_conv = dy.conv2d_bias(dy_embedded, conv, bias, (1,1,1,1), is_valid=False)

        dy_out = dy.sum_elems(dy_conv)
        gold = dy.scalarInput(0)

        dy_loss = dy.square(gold - dy_out)

        dy_loss.npvalue()
        dy_loss.backward()
        trainer.update()

    ends = [embed.npvalue(), conv.npvalue(), bias.npvalue()]
    return starts, ends, dy_loss.npvalue()

if __name__ == "__main__":
    man = mp.Manager()
    return_dict = man.dict()
    gpu_job = mp.Process(target=gpu, args=(return_dict,))
    cpu_job = mp.Process(target=cpu, args=(return_dict,))
    gpu_job.start()
    cpu_job.start()
    gpu_job.join()
    cpu_job.join()
    for c, g in zip(return_dict['cpu_start'], return_dict['gpu_start']):
        np.testing.assert_allclose(c, g)
    print("All Starting weights are equal!")
    for c, g, name in zip(return_dict['cpu'], return_dict['gpu'], ['embed', 'conv', 'bias']):
        try:
            np.testing.assert_allclose(c, g)
        except Exception as e:
            print(name)
            print(e)
            print(c)
            print(g)
    print("CPU Loss: {}".format(return_dict['cpu_loss']))
    print("GPU Loss: {}".format(return_dict['gpu_loss']))
