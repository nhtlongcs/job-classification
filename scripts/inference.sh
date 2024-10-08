CONFIG_ARG=$1

if [ ! -f $CONFIG_ARG ]; then
    echo "Config file not found!"
    exit 1
fi

python preprocess.py -c $CONFIG_ARG
python retrieve.py -c $CONFIG_ARG
python rerank.py -c $CONFIG_ARG
python inference.py -c $CONFIG_ARG
python submission.py -c $CONFIG_ARG