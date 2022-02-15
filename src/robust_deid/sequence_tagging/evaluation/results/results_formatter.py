import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

class ResultsFormatter(object):
    
    @staticmethod
    def get_results_df(results):
        def change_column_names(group):
            group.rename(columns=lambda name: re.sub('(.*_)(?=[a-zA-Z0-9]+$)', '', name), inplace=True)
            return group
        results_df = pd.DataFrame([results])
        group_pattern = '(.*(?=_recall|_precision|_f1|_number))'
        grouped = results_df.groupby(results_df.columns.str.extract(group_pattern, expand=False), axis=1)
        grouped_df_dict = {name:change_column_names(group) for name, group in grouped}
        grouped_df = pd.concat(grouped_df_dict.values(), axis=1, keys=grouped_df_dict.keys())
        return grouped_df.T.unstack().droplevel(level=0, axis=1)[['precision', 'recall', 'f1', 'number']]
    
    @staticmethod
    def get_confusion_matrix(confusion_matrix, ner_types):
        S = 15
        normalize = True
        title = 'Confusion Matrix'
        cmap=plt.cm.Blues
        classes = ner_types + ['O', ]
        plt.figure(figsize = (S, S))
        
        cm = confusion_matrix
        cmbk = cm
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(S, S*0.8))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(0,cm.shape[1]),
               yticks=np.arange(0,cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='Ground Truth',
               xlabel='Predicted')
        ax.xaxis.get_label().set_fontsize(16)
        ax.yaxis.get_label().set_fontsize(16)
        ax.title.set_size(16)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 14)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'#'.2f'
        fmt='d'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cmbk[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",fontsize=12)
        fig.tight_layout()
        return fig