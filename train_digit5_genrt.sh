data_conf=configs/datasets/da/digit5_nflow.yaml
conf=configs/trainers/ssl/fixmatch_ema/digit5.yaml
trainer=FixMatchNFlowClassMixConsistencyDigit5
opt='MODEL.BACKBONE.NAME cnn_digit5_m3sda_nflow_cmix TRAINER.CALOSS.WEIGHT_CON .5 TRAINER.DDAIG.WARMUP 9 '
for((i=0;i<=4;i++));do
GPU=0
CUDA_VISIBLE_DEVICES=$GPU python tools/train.py --root ../../data --trainer $trainer \
	--source-domains mnist usps svhn syn --target-domains mnist_m  \
	--dataset-config-file ${data_conf} --config-file ${conf}  \
	--output-dir output/nflow_cmix_digit5/mnist_m --resume output/nflow_cmix_digit5/usps/nomodel  \
	$opt 2>&1|tee output/nflow_cmix_digit5/fm_nflow_mnistm_${i}.log &
((GPU=GPU+1))
CUDA_VISIBLE_DEVICES=$GPU python tools/train.py --root ../../data --trainer $trainer \
	--source-domains mnist mnist_m svhn syn --target-domains usps  \
	--dataset-config-file ${data_conf} --config-file ${conf}  \
	--output-dir output/nflow_cmix_digit5/usps --resume output/nflow_cmix_digit5/usps/nomodel  \
	$opt 2>&1|tee output/nflow_cmix_digit5/fm_nflow_usps_${i}.log &
((GPU=GPU+1))
CUDA_VISIBLE_DEVICES=$GPU python tools/train.py --root ../../data --trainer $trainer \
	--source-domains usps mnist_m svhn syn --target-domains mnist  \
	--dataset-config-file ${data_conf} --config-file ${conf}  \
	--output-dir output/nflow_cmix_digit5/mnist --resume output/nflow_cmix_digit5/usps/nomodel  \
	$opt 2>&1|tee output/nflow_cmix_digit5/fm_nflow_mnist_${i}.log &
((GPU=GPU+1))
CUDA_VISIBLE_DEVICES=$GPU python tools/train.py --root ../../data --trainer $trainer \
	--source-domains mnist mnist_m usps syn --target-domains svhn  \
	--dataset-config-file ${data_conf} --config-file ${conf}  \
	--output-dir output/nflow_cmix_digit5/svhn --resume output/nflow_cmix_digit5/usps/nomodel  \
	$opt 2>&1|tee output/nflow_cmix_digit5/fm_nflow_svhn_${i}.log &
((GPU=GPU+1))
CUDA_VISIBLE_DEVICES=$GPU python tools/train.py --root ../../data --trainer $trainer \
	--source-domains mnist mnist_m svhn usps --target-domains syn  \
	--dataset-config-file ${data_conf} --config-file ${conf}  \
	--output-dir output/nflow_cmix_digit5/syn --resume output/nflow_cmix_digit5/usps/nomodel  \
	$opt 2>&1|tee output/nflow_cmix_digit5/fm_nflow_syn_${i}.log &
done
