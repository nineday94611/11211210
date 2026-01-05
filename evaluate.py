import pandas as pd
from sklearn.metrics import roc_auc_score



pred_df = pd.read_csv('sample_submission.csv')  # pred
true_df = pd.read_csv('test_answer.csv')      # answer
public_private_df = pd.read_csv('test_public_private_id.csv')  

public_ids = public_private_df['public'].to_list()
private_ids  = public_private_df['private'].to_list()

public_pred_df = pred_df[pred_df['unique_id'].isin(public_ids)]
public_true_df = true_df[true_df['unique_id'].isin(public_ids)]
private_pred_df = pred_df[pred_df['unique_id'].isin(private_ids)]
private_true_df = true_df[true_df['unique_id'].isin(private_ids)]

#public
public_auc_gender = roc_auc_score(public_true_df['gender'], public_pred_df['gender'])
public_auc_hold_racket = roc_auc_score(public_true_df['hold racket handed'], public_pred_df['hold racket handed'])


public_true_play_years = pd.get_dummies(public_true_df['play years'], prefix='play years')
public_pred_play_years = public_pred_df[['play years_0', 'play years_1', 'play years_2']]
public_auc_play_years = roc_auc_score(public_true_play_years, public_pred_play_years, average='micro', multi_class='ovr')


public_true_level = pd.get_dummies(public_true_df['level'], prefix='level')
public_pred_level = public_pred_df[['level_2', 'level_3', 'level_4', 'level_5']]
public_auc_level = roc_auc_score(public_true_level, public_pred_level, average='micro',  multi_class='ovr')

print('public:')
print(f'Gender AUC: {public_auc_gender:.4f}')
print(f'Hold Racket Handed AUC: {public_auc_hold_racket:.4f}')
print(f'Play Years AUC: {public_auc_play_years:.4f}')
print(f'Level AUC: {public_auc_level:.4f}')
print(f'Total AUC: {0.25*(public_auc_gender+public_auc_hold_racket+public_auc_play_years+public_auc_level):.4f}')

#private
private_auc_gender = roc_auc_score(private_true_df['gender'], private_pred_df['gender'])
private_auc_hold_racket = roc_auc_score(private_true_df['hold racket handed'], private_pred_df['hold racket handed'])


private_true_play_years = pd.get_dummies(private_true_df['play years'], prefix='play years')
private_pred_play_years = private_pred_df[['play years_0', 'play years_1', 'play years_2']]
private_auc_play_years = roc_auc_score(private_true_play_years, private_pred_play_years, average='micro', multi_class='ovr')


private_true_level = pd.get_dummies(private_true_df['level'], prefix='level')
private_pred_level = private_pred_df[['level_2', 'level_3', 'level_4', 'level_5']]
private_auc_level = roc_auc_score(private_true_level, private_pred_level, average='micro',  multi_class='ovr')

print('private:')
print(f'Gender AUC: {private_auc_gender:.4f}')
print(f'Hold Racket Handed AUC: {private_auc_hold_racket:.4f}')
print(f'Play Years AUC: {private_auc_play_years:.4f}')
print(f'Level AUC: {private_auc_level:.4f}')
print(f'Total AUC: {0.25*(private_auc_gender+private_auc_hold_racket+private_auc_play_years+private_auc_level):.4f}')