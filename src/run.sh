conda activate env
python train.py --pretrained_model your_model_path
python test.py
python post_processing.py
python eval.py -d tc -r ../results/DBG_reduce_dim-0627-1512-tianchi-validation.json