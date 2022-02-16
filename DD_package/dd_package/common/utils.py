import os
import sys
import pickle

import sklearn.metrics.mean_absolute_error as mae

import sklearn.metrics.mean_squared_error as mse



def save_a_dict(a_dict, name, save_path, ):
    with open(os.path.join(save_path, name+".pickle"), "wb") as fp:
        pickle.dump(a_dict, fp)
    return None


def load_a_dict(name, save_path, ):
    with open(os.path.join(save_path, name + ".pickle"), "rb") as fp:
        a_dict = pickle.load(fp)
    return a_dict


def print_the_evaluated_results(specifier, learning_method, ):

    with open("../results/" + specifier, "rb") as fp:
        results = pickle.load(fp)

    # Regression metrics
    MEA, RMSE, MRAE, JSD, R2_Score, MEAPE_mu, MEAPE_std = [], [], [], [], [], [], []
    # Classification and clustering metrics
    ARI, NMI, Precision, Recall, F1_Score, ROC_AUC, ACC = [], [], [], [], [], [], []

    for repeat, result in results.items():
        y_true = result["y_test"]
        y_pred = result["y_pred"]

        if learning_method == "regression":  # or learning_method == "baseline":

            MEA.append(mae(y_true=y_true, y_pred=y_pred))
            RMSE.append(rmse(y_true=y_true, y_pred=y_pred))
            MRAE.append(mrae(y_true=y_true, y_pred=y_pred))
            JSD.append(jsd(y_true=y_true, y_pred=y_pred).mean())
            R2_Score.append(metrics.r2_score(y_true, y_pred))
            meape_errors = mean_estimation_absolute_percentage_error(
                y_true=y_true, y_pred=y_pred, n_iters=100)

            MEAPE_mu.append(meape_errors.mean(axis=0))
            MEAPE_std.append(meape_errors.std(axis=0))

        else:
            ARI.append(metrics.adjusted_rand_score(y_true, y_pred))
            NMI.append(metrics.normalized_mutual_info_score(y_true, y_pred))
            JSD.append(jsd(y_true=y_true, y_pred=y_pred).mean())
            Precision.append(metrics.precision_score(y_true, y_pred, average='weighted'))
            Recall.append(metrics.recall_score(y_true, y_pred, average='weighted'))
            F1_Score.append(metrics.f1_score(y_true, y_pred, average='weighted'))
            ROC_AUC.append(metrics.roc_auc_score(y_true, y_pred, average='weighted'))
            meape_errors = mean_estimation_absolute_percentage_error(
                y_true=y_true, y_pred=y_pred, n_iters=100)

            MEAPE_mu.append(meape_errors.mean(axis=0))
            MEAPE_std.append(meape_errors.std(axis=0))
            ACC.append(metrics.accuracy_score(y_true, y_pred, ))

    if learning_method == "regression":
        MEA = np.nan_to_num(np.asarray(MEA))
        RMSE = np.nan_to_num(np.asarray(RMSE))
        MRAE = np.nan_to_num(np.asarray(MRAE))
        JSD = np.nan_to_num(np.asarray(JSD))
        R2_Score = np.nan_to_num(np.asarray(R2_Score))
        MEAPE_mu = np.nan_to_num(np.asarray(MEAPE_mu))

        mae_ave = np.mean(MEA, axis=0)
        mae_std = np.std(MEA, axis=0)

        rmse_ave = np.mean(RMSE, axis=0)
        rmse_std = np.std(RMSE, axis=0)

        mrae_ave = np.mean(MRAE, axis=0)
        mrae_std = np.std(MRAE, axis=0)

        jsd_ave = np.mean(JSD, axis=0)
        jsd_std = np.std(JSD, axis=0)

        r2_ave = np.mean(R2_Score, axis=0)
        r2_std = np.std(R2_Score, axis=0)

        meape_ave = np.mean(MEAPE_mu, axis=0)
        meape_std = np.std(MEAPE_mu, axis=0)

        print("   mae ", " \t rmse ", "\t mrae",
              "\t r2_score ", "\t meape ", "\t jsd ",
              )

        print(" Ave ", " std", " Ave ", " std ", " Ave ", " std ", " Ave ", " std ",
              " Ave ", " std ", " Ave ", " std ",
              )

        print(
            "%.3f" % mae_ave, "%.3f" % mae_std,
            "%.3f" % rmse_ave, "%.3f" % rmse_std,
            "%.3f" % mrae_ave, "%.3f" % mrae_std,
            "%.3f" % r2_ave, "%.3f" % r2_std,
            "%.3f" % meape_ave, "%.3f" % meape_std,
            "%.3f" % jsd_ave, "%.3f" % jsd_std,
        )

    else:

        JSD = np.nan_to_num(np.asarray(JSD))
        MEAPE_mu = np.nan_to_num(np.asarray(MEAPE_mu))
        ARI = np.nan_to_num(np.asarray(ARI))
        NMI = np.nan_to_num(np.asarray(NMI))
        Precision = np.nan_to_num(np.asarray(Precision))
        Recall = np.nan_to_num(np.asarray(Recall))
        F1_Score = np.nan_to_num(np.asarray(F1_Score))
        ROC_AUC = np.nan_to_num(np.asarray(ROC_AUC))
        ACC = np.nan_to_num((np.asarray(ACC)))

        ari_ave = np.mean(ARI, axis=0)
        ari_std = np.std(ARI, axis=0)

        nmi_ave = np.mean(NMI, axis=0)
        nmi_std = np.std(NMI, axis=0)

        precision_ave = np.mean(Precision, axis=0)
        precision_std = np.std(Precision, axis=0)

        recall_ave = np.mean(Recall, axis=0)
        recall_std = np.std(Recall, axis=0)

        f1_score_ave = np.mean(F1_Score, axis=0)
        f1_score_std = np.std(F1_Score, axis=0)

        roc_auc_ave = np.mean(ROC_AUC, axis=0)
        roc_auv_std = np.std(ROC_AUC, axis=0)

        jsd_ave = np.mean(JSD, axis=0)
        jsd_std = np.std(JSD, axis=0)

        meape_ave = np.mean(MEAPE_mu, axis=0)
        meape_std = np.std(MEAPE_std, axis=0)

        acc_ave = np.mean(ACC, axis=0)
        acc_std = np.std(ACC, axis=0)

        print("  ari ", "  nmi ", "\t preci", "\t recall ",
              "\t f1_score ", "\t roc_auc ", "\t meape ", "\t jsd ", "\t acc"
              )

        print(" Ave ", " std", " Ave ", " std ", " Ave ", " std ", " Ave ", " std ",
              " Ave ", " std ", " Ave ", " std ", " Ave ", " std ", " Ave ", " std ", " Ave ", " std "
              )

        print("%.3f" % ari_ave, "%.3f" % ari_std,
              "%.3f" % nmi_ave, "%.3f" % nmi_std,
              "%.3f" % precision_ave, "%.3f" % precision_std,
              "%.3f" % recall_ave, "%.3f" % recall_std,
              "%.3f" % f1_score_ave, "%.3f" % f1_score_std,
              "%.3f" % roc_auc_ave, "%.3f" % roc_auv_std,
              "%.3f" % meape_ave, "%.3f" % meape_std,
              "%.3f" % jsd_ave, "%.3f" % jsd_std,
              "%.3f" % acc_ave, "%.3f" % acc_std,
              )

    return None
