source activate env
python train.py
python test.py
python post_processing.py
python eval.py --gtfile ../data/ActivityNet/video_info_19993.json --evalfile ../results/DBG-0623-1614-activitynet-validation.json