# CS-QCFS: Bridging the performance gap in ultra-low latency spiking neural networks

Codes for CS-QCFS: Bridging the performance gap in ultra-low latency spiking neural networks

## Usage

Please first change the variable "DIR" at File ".Preprocess\getdataloader.py", line 9 to your own dataset directory

Train model with QCFS-Layer

```bash
python main.py train --bs=BATACHSIZE --model={vgg16, resnet18, resnet50, resnet101} --data={cifar10, cifar100} --id=YOUR_MODEL_NAME --l=QUANTIZATION_STEP --seed=YOUR_SEED --acitvation_mode={origin,softplus}
```

Train model with CSQCFS-Layer

```bash
python main_finetune.py train --bs=BATACHSIZE --model={vgg16, resnet18, resnet50, resnet101} --data={cifar10, cifar100} --l=QUANTIZATION_STEP --seed=YOUR_SEED --acitvation_mode={origin,softplus} --layer_id=YOUR_QCFS_MODEL_NAME --id=YOUR_MODEL_NAME --channel_num YOUR_CSQCFS_LAYER_NUM
```

Test accuracy in ann mode or snn mode

```bash
python main.py test --bs=BATACHSIZE --model={vgg16, resnet18} --data={cifar10, cifar100} --id=YOUR_MODEL_NAME --mode={ann, snn} --t=SIMULATION_TIME --presim_len 4 --acitvation_mode={origin,softplus} --channel_num YOUR_CSQCFS_LAYER_NUM
```
