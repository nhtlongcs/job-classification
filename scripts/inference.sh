STELLA_CFG="configs/stella_bm25_mistralnemo.ini"
GEMMA2_CFG="configs/gemma2_bm25_mistralnemo.ini"

STELLA_CFG_GEMMA="configs/stella_bm25_gemma2.ini"
GEMMA2_CFG_GEMMA="configs/gemma2_bm25_gemma2.ini"

python preprocess.py -c $STELLA_CFG # only need to preprocess once

python src/retrieve.py -c $STELLA_CFG
python src/retrieve.py -c $GEMMA2_CFG

python src/rerank.py -c $STELLA_CFG
python src/rerank.py -c $GEMMA2_CFG

python src/inference_batch.py -c $STELLA_CFG 
python src/inference_batch.py -c $GEMMA2_CFG 

python src/inference_batch.py -c $STELLA_CFG_GEMMA
python src/inference_batch.py -c $QWEN_CFG_GEMMA
python src/inference_batch.py -c $ICL_CFG_GEMMA
python src/inference_batch.py -c $GEMMA2_CFG_GEMMA

python src/submission.py -c $STELLA_CFG_GEMMA -o artifacts/results_stella_gemma2/
python src/submission.py -c $GEMMA2_CFG_GEMMA -o artifacts/results_gemma2_gemma2/

python src/submission.py -c $STELLA_CFG-o artifacts/results_stella_mistral/
python src/submission.py -c $GEMMA2_CFG-o artifacts/results_gemma2_mistral/

python merge.py --ensemble_files artifacts/results_stella_gemma2/classification.csv \
                artifacts/results_gemma2_gemma2/classification.csv \
                artifacts/results_stella_mistral/classification.csv \
                artifacts/results_gemma2_mistral/classification.csv \
                --compare_file latest_predictions.csv