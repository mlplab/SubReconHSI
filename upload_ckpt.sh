# !/usr/bin/zsh


CMDNAME=`basename $0`


dataset="CAVE"
start_time="0000"


while getopts b:d:c:m:bn: OPT
do
    echo "$OPTARG"
    case $OPT in
        d) datasets=$OPTARG ;;
        s) start_time=$OPTARG ;;
        *) echo "Usage: $CMDNAME [-b batch size] [-e epoch]" 1>&2
            exit 1;;
    esac
done
