import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import shap


def legend_labels(labels, shap_values):
    return ['{}({:.3f})'.format(label, shap_value) for label, shap_value in zip(labels, shap_values)]


class Shap:

    def __init__(self, input_df, models, pdb_names, model_type):
        # monkey patch
        self.shap_values = np.zeros((len(input_df), len(input_df.columns)), np.float)
        self.input_df = input_df
        self.shap_return_value = None
        self.model_type = model_type
        self.base_value = 0.
        self.pdb_names = pdb_names

        for model in models:
            if model_type=='xgboost':
                # monkey patch
                booster = model.get_booster() 
                model_bytearray = booster.save_raw()[4:]
                booster.save_raw = lambda : model_bytearray
                self.explainer = shap.TreeExplainer(model=booster, feature_dependence='tree_path_dependent', model_output='margin')
                self.shap_values += np.array(self.explainer.shap_values(X=input_df))/len(models)
                self.base_value += self.explainer.expected_value/len(models)

            elif model_type=='lightgbm':
                booster = model
                # shap expaliner
                self.explainer = shap.TreeExplainer(model=booster, feature_dependence='tree_path_dependent', model_output='margin')
                # for class1
                self.shap_values += self.explainer.shap_values(X=input_df)[1]/len(models) # for class1
                self.base_value += self.explainer.expected_value[1]/len(models)
            
            elif model_type=='svm':
                booster = model
                # shap expaliner
                self.explainer = shap.KernelExplainer(model.predict, input_df)
                self.shap_values += self.explainer.shap_values(input_df)/len(models)
                self.base_value += self.explainer.expected_value/len(models)



    def summary_plot(self, out_path, plot_type, show=False):
        # summary plot
        shap.summary_plot(
                    shap_values=self.shap_values, 
                    features=self.input_df, 
                    feature_names=self.input_df.columns,
                    plot_type=plot_type,
                    show=show)
        pylab.tight_layout()
        plt.savefig(out_path)
        plt.close()


    def decision_plot(self, out_path, show=False):
        # decision plot

        self.shap_return_value = shap.decision_plot(
                                    base_value= self.base_value,
                                    shap_values=self.shap_values,
                                    features=self.input_df,
                                    feature_names=list(self.input_df.columns),
                                    return_objects=True,
                                    feature_order='hclust',
                                    # link='logit',
                                    show=show)

        # save plot
        pylab.tight_layout()
        plt.savefig(out_path)
        plt.close()


    def decision_ok_vs_miss_plot(self, y_pred, y_test, out_path, show=False):
        # decision plot
        misclassified = np.where(y_pred < 0, 0, np.round(y_pred).astype(int)) != y_test
        shap.decision_plot(
                        base_value=self.base_value, 
                        shap_values=self.shap_values,
                        features=self.input_df, 
                        feature_names=list(self.input_df.columns),
                        feature_order=self.shap_return_value.feature_idx,
                        xlim=self.shap_return_value.xlim,
                        highlight=misclassified,
                        show=show)
        # save plot
        pylab.tight_layout()
        plt.savefig(out_path)
        plt.close()
        

    def decision_miss_data_plot(self, y_pred, y_test, out_path, show=False):
        # only miss data plot
        misclassified = np.where(y_pred < 0, 0, np.round(y_pred).astype(int)) != y_test
        print('misclassified: ', misclassified)
        print('pdb_names: ', self.pdb_names[misclassified])
        shap.decision_plot(
                        base_value=self.base_value, 
                        shap_values=self.shap_values[misclassified],
                        features=self.input_df.iloc[misclassified], 
                        feature_names=list(self.input_df.columns),
                        feature_order=self.shap_return_value.feature_idx,
                        xlim=self.shap_return_value.xlim,
                        highlight=np.arange(len(misclassified[misclassified==True])),
                        legend_labels= legend_labels(self.pdb_names[misclassified], self.base_value+self.shap_values[misclassified].sum(axis=1)),
                        legend_location='lower right',
                        show=show)
        # save plot
        pylab.tight_layout()
        plt.savefig(out_path)
        plt.close()

    
    def decision_high_prob_data_plot(self, y_pred, prob_threshold, out_path, show=False):
        # high probability data plot
        ok_high_prob_idxs = y_pred >= prob_threshold
        # print(y_pred)

        shap.decision_plot(
                        base_value=self.base_value, 
                        shap_values=self.shap_values[ok_high_prob_idxs],
                        features=self.input_df[ok_high_prob_idxs], 
                        feature_names=list(self.input_df.columns),
                        feature_order=self.shap_return_value.feature_idx,
                        xlim=self.shap_return_value.xlim,
                        legend_labels= legend_labels(self.pdb_names[ok_high_prob_idxs], self.base_value+self.shap_values[ok_high_prob_idxs].sum(axis=1)),
                        legend_location='lower right',
                        show=show)
        # save plot
        pylab.tight_layout()
        plt.savefig(out_path)
        plt.close()


    def dependence_plot(self, ind, interaction_index, out_path, show=False):
        # dependence_plot
        shap.dependence_plot(
                        ind=ind,
                        interaction_index=interaction_index,
                        shap_values=self.shap_values,
                        features=self.input_df,
                        feature_names=self.input_df.columns,
                        show=show)
        # save plot
        pylab.tight_layout()
        plt.savefig(out_path)
        plt.close()


    def force_plot(self, y_pred, y_test, out_path, show=False):
        # force plot
        misclassified = np.where(y_pred < 0, 0, np.round(y_pred).astype(int)) != y_test
        miss_index = np.argsort(misclassified)[::-1][0]
        
        shap.force_plot(
                    base_value=self.base_value, 
                    shap_values=self.shap_values[miss_index],
                    features=self.input_df.iloc[miss_index],
                    matplotlib=True,
                    show=show)
        # save plot
        pylab.tight_layout()
        plt.savefig(out_path)
        plt.close()