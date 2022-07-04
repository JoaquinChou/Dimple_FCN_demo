python train_supcon.py --batch_size 32 --learning_rate 0.5 --temp 0.1 --cosine

# plot tsne
python gen_label_feature_dimple.py --model_path supcon_models_02-18-19-22/last.pth --results_txt 02-18-19-22_last
python gen_label_feature_dimple.py --model_path supcon_models_05-04-09-33/last.pth --results_txt 05-04-09-33_last --isPrintImgListName True --img_list_name_txt last_img_name
python TSNE.py --initial_dims 512 --results_txt 02-18-19-22_last
python TSNE.py --initial_dims 512 --results_txt 05-04-09-33_last --output_anormal True --img_path_name last_img_name.txt

# 二阶段训练
python train.py

