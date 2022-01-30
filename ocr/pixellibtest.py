from pixellib.custom_train import instance_custom_training

vis_img = instance_custom_training()
vis_img.load_dataset("../datasets/Manga109/labelme")
vis_img.visualize_sample()
