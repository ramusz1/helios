import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

from .HyperTools import compile_results
from .util import load_rects, HOTDataset, HOTDatasetMultiCam

def eval_results(dataset: HOTDataset, dataset_type, sequence_names, models, plot):

    predictions = f"../outputs/model_predictions"
    camera_type = dataset.camera_type

    auc = np.zeros((len(sequence_names), len(models)))
    dp_20 = np.zeros_like(auc)

    for j, model in enumerate(models):
        for i, sequence_name in enumerate(sequence_names):
            scene = dataset.get_scene(sequence_name)
            pred_file = os.path.join(predictions, model, dataset_type, camera_type, f"{sequence_name}.txt")
            if os.path.exists(pred_file):
                
                y_pred = load_rects(pred_file)
                if len(y_pred) != len(scene.y_true):
                    print(sequence_name, model, len(y_pred), len(scene.y_true), len(scene.falsecolor))
                _, _, _, auc[i,j], dp_20[i,j] = compile_results(scene.y_true, y_pred)
            else:
                auc[i,j] = 0
                dp_20[i,j] = 0

    # find the best model
    best_model_id = np.argmax(np.mean(auc, axis=0))
    # create a dataframe
    df_auc = pd.DataFrame(data=auc, index=sequence_names, columns=[m + "_auc" for m in models])
    df_dp20 = pd.DataFrame(data=dp_20, index=sequence_names, columns=[m + "_dp20" for m in models])
    df = pd.concat((df_auc, df_dp20), axis=1)
    df = df.sort_values(models[best_model_id] + "_auc", ascending=True)
    df_auc = df[[m + "_auc" for m in models]]
    df_dp20 = df[[m + "_dp20" for m in models]]
    # plot results
    if plot:
        fig, axs = plt.subplots(1,2,figsize=(12,9))
        df_auc.plot.barh(rot=0, ax=axs[0])
        df_dp20.plot.barh(rot=0, ax=axs[1])
        axs[0].set_xlim(0,1)
        axs[1].set_xlim(0,1)
        fig.tight_layout()

    summary = df.mean().reset_index(name="value")
    summary["model"] = models + models
    summary["metric"] = ["mean AUC"] * len(models) + ["mean DP20"] * len(models)
    summary = summary.drop(columns=["index"])
    summary = summary.pivot_table(values="value", index="model", columns=["metric"]).reset_index()
    summary = summary.sort_values("mean AUC", ascending=False)
    return df, summary

def eval_results_v2(dataset: HOTDatasetMultiCam, models, plot, sequence_ids=None):

    predictions = f"../outputs/model_predictions"

    if sequence_ids is None:
        sequence_ids = np.arange(len(dataset))

    auc = np.zeros((len(sequence_ids), len(models)))
    dp_20 = np.zeros_like(auc)
    
    sequence_names = [dataset[s].name for s in sequence_ids]

    missing_scenes = []
    for j, model in enumerate(models):
        for i, scene in enumerate(sequence_ids):
            scene = dataset[scene]
            pred_file = os.path.join(predictions, model, dataset.dataset_type, scene.camera_type, scene.name + ".txt")
            if os.path.exists(pred_file):
                y_pred = load_rects(pred_file)
                if len(y_pred) != len(scene.y_true):
                    print(scene.camera_type, scene.name, model, len(y_pred), len(scene.y_true), len(scene.falsecolor))
                    assert False
                _, _, _, auc[i,j], dp_20[i,j] = compile_results(scene.y_true, y_pred)
            else:
                missing_scenes.append((scene.name, scene.camera_type))
                auc[i,j] = 0
                dp_20[i,j] = 0

    if len(missing_scenes) > 0:
        print("Predictions for the following scenes were not found for some models")
        print(missing_scenes)

    # find the best model
    best_model_id = np.argmax(np.mean(auc, axis=0))
    # create a dataframe
    df_auc = pd.DataFrame(data=auc, index=sequence_names, columns=[m + "_auc" for m in models])
    df_dp20 = pd.DataFrame(data=dp_20, index=sequence_names, columns=[m + "_dp20" for m in models])
    df = pd.concat((df_auc, df_dp20), axis=1)
    df = df.sort_values(models[best_model_id] + "_auc", ascending=True)
    df_auc = df[[m + "_auc" for m in models]]
    df_dp20 = df[[m + "_dp20" for m in models]]
    # plot results
    if plot:
        fig, axs = plt.subplots(1,2,figsize=(12,9))
        df_auc.plot.barh(rot=0, ax=axs[0])
        df_dp20.plot.barh(rot=0, ax=axs[1])
        axs[0].set_xlim(0,1)
        axs[1].set_xlim(0,1)
        fig.tight_layout()

    summary = df.mean().reset_index(name="value")
    summary["model"] = models + models
    summary["metric"] = ["mean AUC"] * len(models) + ["mean DP20"] * len(models)
    summary = summary.drop(columns=["index"])
    summary = summary.pivot_table(values="value", index="model", columns=["metric"]).reset_index()
    summary = summary.sort_values("mean AUC", ascending=False)
    return df, summary
