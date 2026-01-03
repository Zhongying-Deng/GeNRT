trainer=FixMatchNFlowClassMixConsistency
opt='MODEL.BACKBONE.NAME resnet18_nflow_cmix TRAINER.CALOSS.WEIGHT_CON .5 TRAINER.DDAIG.WARMUP 9 TRAIN.CHECKPOINT_FREQ 10 OPTIM.MAX_EPOCH 90'
outpath=pacs_genrt
for((i=0;i<=4;i++));
do
GPU=0
 python tools/train.py --root ./data/ --trainer ${trainer} \
 --source-domains art_painting cartoon sketch --target-domains photo \
 --dataset-config-file configs/datasets/da/pacs.yaml \
 --config-file configs/trainers/da/pacs_staged_lr.yaml \
 --output-dir output/${outpath}/photo \
 --resume output/adaptkernel_pacs/sketch/nomodel \
 $opt TRAINER.FEATMIX.CONFIG '(1, 0, 0, 0, 0)' 2>&1|tee output/${outpath}/fm_featmix_photo_${i}.log 
((GPU=GPU+1))
 python tools/train.py --root ./data/ --trainer ${trainer} \
 --source-domains art_painting cartoon photo --target-domains sketch \
 --dataset-config-file configs/datasets/da/pacs.yaml \
 --config-file configs/trainers/da/pacs_staged_lr.yaml \
 --output-dir output/${outpath}/sketch \
 --resume output/adaptkernel_pacs/sketch/nomodel \
 $opt TRAINER.FEATMIX.CONFIG '(1, 0, 0, 0, 0)'  2>&1|tee output/${outpath}/fm_featmix_sketch_${i}.log 
((GPU=GPU+1))
 python tools/train.py --root ./data/ --trainer ${trainer} \
 --source-domains photo cartoon sketch --target-domains art_painting \
 --dataset-config-file configs/datasets/da/pacs.yaml \
 --config-file configs/trainers/da/pacs_staged_lr.yaml \
 --output-dir output/${outpath}/art_painting \
 --resume output/adaptkernel_pacs/sketch/nomodel \
 $opt TRAINER.FEATMIX.CONFIG '(1, 0, 0, 0, 0)' 2>&1|tee output/${outpath}/fm_featmix_art_${i}.log
((GPU=GPU+1))
 python tools/train.py --root ./data/ --trainer ${trainer} \
 --source-domains art_painting photo sketch --target-domains cartoon \
 --dataset-config-file configs/datasets/da/pacs.yaml \
 --config-file configs/trainers/da/pacs_staged_lr.yaml \
 --output-dir output/${outpath}/cartoon \
 --resume output/adaptkernel_pacs/sketch/nomodel \
 $opt TRAINER.FEATMIX.CONFIG '(1, 0, 0, 0, 0)' 2>&1|tee output/${outpath}/fm_featmix_cartoon_${i}.log
done
