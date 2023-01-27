from ..utils import argObj, train_validation_partition_images, make_data_loaders

DATA_DIR='/home/maria/Algonauts2023'
PARENT_SUBMISSION_DIR='/home/maria/Algonauts2023_submission'


def vanilla_alexnet_pipeline(subj, parent_submission_dir=PARENT_SUBMISSION_DIR, data_dir=DATA_DIR):
    data_dir='/home/maria/Algonauts2023'
    parent_submission_dir='/home/maria/Algonauts2023_submission'
    args = argObj(data_dir, parent_submission_dir, subj)
    idxs_train, idxs_val, idxs_test=train_validation_partition_images(args, rand_seed=17)
    train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader=make_data_loaders(train_imgs_paths, test_imgs_paths, batch_size=300)


vanilla_alexnet_pipeline()