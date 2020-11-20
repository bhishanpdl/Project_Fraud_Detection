# Files Required by Heroku
- setup.sh
- Procfile
- runtime.txt

# Heroku Limitations
- The total file size must be less than 500 MB.
- We can not use tensorflow/keras model in free tier, the build itself is more than 500 MB.
- We can use usual ML models such as xgboost, lightgbm, catboost, pycaret etc without worrying about size.

Some tips:
- Load the data from github repo Datasets rather than uploading data to github repo/branch.
- Do not install unwanted modules.
- Always use version number in requirements.txt (eg. shap < 0.36 does not support shap getjs)
