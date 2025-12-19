cd ../
python main.py \
    --gpu 1 \
    --seed 10 \
    --results_path /root/3D/BridgeNet-github-refactor/results \
    --test_mode ckpt \
  net \
    -b wideresnet50 \
    -le layer2 \
    -le layer3 \
    --pretrain_embed_dimension 1536 \
    --target_embed_dimension 1536 \
    --patchsize 3 \
    --meta_epochs 200 \
    --eval_epochs 1 \
    --dsc_layers 3 \
    --dsc_hidden 1536 \
    --pre_proj 1 \
    --noise 0.02 \
    --step 20 \
    --limit 392 \
  dataset \
    --mean 0.5 \
    --std 0.3 \
    --fg 1 \
    --rand_aug 1 \
    --batch_size 4 \
    --imagesize 576 \
    -d cable_gland \
    mvtec \
    /root/3D/GLASS-mvtec-3d-dataset/datasets/mvtec_process \
    /root/3D/GLASS-mvtec-3d-dataset/datasets/dtd/images
