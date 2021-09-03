# !/usr/bin/zsh


CMDNAME=`basename $0`


batch_size=64
# search_epoch=100
train_epoch=150
datasets=("CAVE" "Harvard")
base_model_name="SpectralFusion"
block_num=3
concats=('False' 'True')
modes=("both" "inputOnly" "outputOnly")
loss_modes=("fusion" "fusion" "mse")
start_time=$(date "+%m%d")
# start_time='0702'


while getopts b:e:d:c:m:bn: OPT
do
    echo "$OPTARG"
    case $OPT in
        b) batch_size=$OPTARG ;;
        e) train_epoch=$OPTARG ;;
        d) datasets=$OPTARG ;;
        c) concat=$OPTARG ;;
        m) model_name=$OPTARG ;;
        bn) block_num=$OPTARG ;;
        *) echo "Usage: $CMDNAME [-b batch size] [-e epoch]" 1>&2
            exit 1;;
    esac
done


for dataset in $datasets; do
    skicka mkdir 2021/SpectralFusion/$dataset/ckpt_$start_time/
    for concat in $concats; do
        i=1
        for mode in $modes; do
            echo $dataset $concat $mode $i $loss_modes[$i]
            python train_fusion.py -e $train_epoch -d $dataset -st $start_time -bn $block_num -c $concat -b $batch_size -m $base_model_name -md $mode -l $loss_modes[$i]
            python evaluate_fusion.py -e $train_epoch -d $dataset -st $start_time -bn $block_num -c $concat -b $batch_size -m $base_model_name -md $mode -l $loss_modes[$i]

            model_name=$base_model_name\_0$block_num\_${loss_modes[$i]}\_$mode\_$start_time\_$concat
            skicka mkdir 2021/SpectralFusion/$dataset/ckpt_$start_time/$model_name
            mkdir ../SCI_result/$dataset\_$start_time/$model_name/$model_name\_upload
            cp ../SCI_result/$dataset\_$start_time/$model_name/output.csv ../SCI_result/$dataset\_$start_time/$model_name/$model_name\_upload
            skicka upload ../SCI_ckpt/$dataset\_$start_time/all_trained/$model_name.tar 2021/SpectralFusion/$dataset/ckpt_$start_time/
            skicka upload ../SCI_result/$dataset\_$start_time/$model_name/$model_name\_upload/ 2021/SpectralFusion/$dataset/ckpt_$start_time/$model_name
            let i++
        done
    done
done
