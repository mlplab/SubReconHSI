# !/usr/bin/zsh


CMDNAME=`basename $0`


batch_size=16
# search_epoch=100
train_epoch=150
datasets=("CAVE" "Harvard")
model_names=("HSCNN" "HyperReconNet" "DeepSSPrior")
block_num=9
concat='False'
loss_modes=("mse" "mse_sam")
start_time=$(date "+%m%d")
# start_time='0702'


while getopts b:d:c:m:bn: OPT
do
    echo "$OPTARG"
    case $OPT in
        b) batch_size=$OPTARG ;;
        # e) epoch=$OPTARG ;;
        d) datasets=$OPTARG ;;
        c) concat=$OPTARG ;;
        m) model_name=$OPTARG ;;
        bn) block_num=$OPTARG ;;
        *) echo "Usage: $CMDNAME [-b batch size] [-e epoch]" 1>&2
            exit 1;;
    esac
done


for dataset in $datasets; do
    for loss_mode in $loss_modes; do
        skicka mkdir 2021/HiNAS/$dataset/ckpt_$start_time/SOTA
        for model_name in $model_names; do
            python train_SOTA.py -e $train_epoch -d $dataset -l $loss_mode -st $start_time -bn $block_num -c $concat -b $batch_size -m $model_name
            python evaluate_SOTA.py -e $train_epoch -d $dataset -l $loss_mode -st $start_time -bn $block_num -c $concat -b $batch_size -m $model_name

            # skicka mkdir 2021/HiNAS/$dataset/ckpt_$start_time/sota/$model_name
            model_name=$model_name\_0$block_num\_$loss_mode\_$start_time\_$concat
            mkdir ../SCI_result/$dataset\_sota_$start_time/$model_name/$model_name\_upload
            cp ../SCI_result/$dataset\_sota_$start_time/$model_name/output.csv ../SCI_result/$dataset\_sota_$start_time/$model_name/$model_name\_upload
            skicka upload ../SCI_ckpt/$dataset\_$start_time/all_trained_sota/ 2021/HiNAS/$dataset/ckpt_$start_time/SOTA
            skicka upload ../SCI_result/$dataset\_sota_$start_time/$model_name/$model_name\_upload/ 2021/HiNAS/$dataset/ckpt_$start_time/SOTA/$model_name
        done
    done
done
