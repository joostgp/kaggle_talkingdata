# kaggle_talkingdata
TalkingData Mobile User Demographics (Kaggle competition)

## Step 1: create cross-validation datasets
create_cv_set.ipynb

## Step 2: create feature sets
- Related to brand and device model: analysis_features_brand.ipynb
- Related to app_id: analysis_features_appid.ipynb
- Related to app_labels: analysis_features_app_labels.ipynb
- Related to event counts: analysis_features_events.ipynb
- Related to geographical data: analysis_features_geo.ipynb
- Related to grouped labels: analysis_features_group.ipynb

### Optionally: do feature selection:
The feature set consists of 200k features, many of them of correlated and have very low variance. The following scripts can be used to select the optimal feature set:
- analysis_feature_selection

## Train models

### Train Level 0 models: 
65% of the samples do not contain any information about events. Level 0 models only use brand and device model information. Four models are used:
- Bayesian model: train_model_0_bayesian.ipynb
- Logistic model: train_model_0_logistic.ipynb
- XGBoost model: train_model_0_xgboost.ipynb
- Keras Neural Net model: train_model_0_nn.ipynb

### Train Level 1 models:
Level 1 models use the brand and device model features and features based on installed apps and labels of installed apps and are trained on all data. Three models are used:
- XGBoost model (gblinear): not uploaded yet
- Logistic model: not uploaded yet
- Keras Neural network: not uploaded yet

### Train Level 2 models:
Level 2 models use all the available features and are trained on devices with events only. Three models are used:
- XGBoost model: not uploaded yet
- Logistic model: not uploaded yet
- Keras Neural network: not uploaded yet


## Stack and ensemble models
Finally stack and ensemble the various models:
- Level 0: train_model_0_stack.ipynb
- Level 0 & 1: train_model_1_stack.ipynb
- Level 0, 1 & 2: train_model_2_stack.ipynb
