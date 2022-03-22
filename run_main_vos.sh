seeds="1 2 3 4"
for seed in $seeds
do
    if [ $seed == 1 ]
    then
        joint=True
        echo $seed $joint
    else
        joint=False
        echo $seed $joint
    fi

#    # RTNet, bs=10
#    python main_vos.py --cfg_file configs/rtnet_davis.json --dataset StylizedDAVIS --checkpoint checkpoints_3models_staticdynamic_cvpr22/model_RX50.pth --batch_size 5 --random_seed $seed --num_workers 2 --joint_encoding $joint
#    
#    # MATNet, bs=30
#    python main_vos.py --dataset StylizedDAVIS --checkpoint checkpoints_3models_staticdynamic_cvpr22/checkpoints_matnet/ --cfg_file configs/matnet_davis.json --random_seed $seed --batch_size 10 --trained_on davis --joint_encoding $joint

    # FusionSeg DAVIS, bs=64
    python main_vos.py --dataset StylizedDAVIS --checkpoint checkpoints_3models_staticdynamic_cvpr22/checkpoints_fseg/latest_twostream_deeplabv3plus_resnet101_davis_os16.pth --cfg_file configs/twostreamv3plus_davis.json --random_seed $seed --batch_size 32 --joint_encoding $joint
    
    # FusionSeg TaoVOS
    python main_vos.py --dataset StylizedDAVIS --checkpoint checkpoints_3models_staticdynamic_cvpr22/checkpoints_fseg/latest_twostream_deeplabv3plus_resnet101_taovos_os16.pth --cfg_file configs/twostreamv3plus_davis.json --random_seed $seed --batch_size 32 --trained_on taovos --joint_encoding $joint
    
    # FusionSeg ImageNet VID
    python main_vos.py --dataset StylizedDAVIS --checkpoint checkpoints_3models_staticdynamic_cvpr22/checkpoints_fseg/latest_twostream_deeplabv3plus_resnet101_imgnetvid_os16.pth --cfg_file configs/twostreamv3plus_davis.json --random_seed $seed --batch_size 32 --trained_on imgnetvid --joint_encoding $joint

    # Best MoCA Model (Recip + Fusion Gated + MATNet BAR)
    #python main_vos.py --dataset StylizedDAVIS --checkpoint /local/riemann/home/msiam/ckpts_cctypes/ckpt_cctype_coatt_gating_recip/MATNet/ --cfg_file configs/cc_types/matnet_davis_bidir_gated.json --random_seed $seed --batch_size 30 --trained_on davis
done
