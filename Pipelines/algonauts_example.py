from ..utils import argObj, partition_indices, make_data_loaders, make_paths, feat_extr_pcs
from torchvision.models.feature_extraction import get_graph_node_names
import mlflow

DATA_DIR='/home/maria/Algonauts2023'
PARENT_SUBMISSION_DIR='/home/maria/Algonauts2023_submission'


def vanilla_alexnet_pipeline(subj, parent_submission_dir=PARENT_SUBMISSION_DIR, data_dir=DATA_DIR):
    data_dir='/home/maria/Algonauts2023'
    parent_submission_dir='/home/maria/Algonauts2023_submission'
    args = argObj(data_dir, parent_submission_dir, subj)
    idxs_train, idxs_val, idxs_test=partition_indices(args, rand_seed=17)
    train_imgs_paths, test_imgs_paths=make_paths(args.train_img_dir, args.test_img_dir)
    train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader=make_data_loaders(train_imgs_paths, test_imgs_paths, batch_size=300)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet')
    train_nodes, _ = get_graph_node_names(model)
    for layer in train_nodes[0:2]:
        features_train, features_val, features_test=feat_extr_pcs(model,layer)
        del model



vanilla_alexnet_pipeline()