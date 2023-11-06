import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

def _mp_fn(index):
    device = xm.xla_device()
    mp_device_loader = pl.MpDeviceLoader(train_loader, device)
    model = MNIST().train().to(device)
    loss_fn = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    for data, target in mp_device_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        xm.optimizer_step(optimizer)

if __name__ == '__main__':
    xmp.spawn(_mp_fn, args=())