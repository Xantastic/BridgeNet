datapath=/root/3D/GLASS-mvtec-3d-dataset/datasets/mvtec_process
augpath=/root/3D/GLASS-mvtec-3d-dataset/datasets/dtd/images
classes=('cookie')
flags=($(for class in "${classes[@]}"; do echo '-d '"${class}"; done))

cd ..
python main.py \
    --gpu 2 \
    --seed 42 \
    --test ckpt \
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
    --distribution 0 \
    --mean 0.5 \
    --std 0.3 \
    --rand_aug 1 \
    --batch_size 4 \
    --resize 224 \
    --imagesize 224 "${flags[@]}" mvtec $datapath $augpath
