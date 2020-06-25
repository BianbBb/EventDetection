source activate env
python train.py --pretrained_model your_model_path
python test.py
python post_processing.py
python eval.py -d tc -r ../results/DBG-0624-1934-tianchi-validation.son