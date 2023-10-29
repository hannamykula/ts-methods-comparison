# Comparison of SoA methods for TS classification applied to TS forecasting
1. Rocket - https://github.com/angus924/rocket/tree/master
   - "rocket-ppv-max" - original implementation of "Rocket"
   - "rocket-avg" - adapted implementation of "Rocket", which uses mean instead of PPV and MAX
2. Minirocket - https://github.com/angus924/minirocket
3. Multirocket - https://github.com/ChangWeiTan/MultiRocket
    - original implementation of "Multirocket", but the classifier was changed to regression model (MLPRegressor)
4. InceptionTime - https://github.com/hfawaz/InceptionTime

The results can be found in the results folder, incl. method, set-up and MAE.