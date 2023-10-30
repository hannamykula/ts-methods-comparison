# Comparison of SoA methods for TS classification applied to TS forecasting
1. Rocket - https://github.com/angus924/rocket/tree/master
   - predictions are done using MLPRegressor
   - "rocket-ppv-max" - original implementation of "Rocket"
   - "rocket-avg" - adapted implementation of "Rocket", which uses mean instead of PPV and MAX
   - also the experiment using mean, std, min, max was conducted, but it performed worse than "rocket-avg" (see table below)
   - the table below contains best constellation of:
     - method chosen from ["Rocket-avg", "Rocket-avg+std", "Rocket-avg+std,min,max"]
     - num. of kernels chosen from [10, 100, 1000, 5000, 10000]
     - lag chosen from [10, 20]
2. Minirocket - https://github.com/angus924/minirocket
   - predictions are done using MLPRegressor
3. Multirocket - https://github.com/ChangWeiTan/MultiRocket
    - original implementation of "Multirocket", but the classifier was changed to regression model (MLPRegressor)
4. InceptionTime - https://github.com/hfawaz/InceptionTime
   - trained for 500 epochs
   - data was normalized using StandardScaler
   - original implementation of "InceptionTime", but the output layer was changed to the one suitable for regression:
   ```
   output_layer = keras.layers.Dense(1, activation='linear')(gap_layer)
   ```


In the main.py you can find the code to run all experiments named above for univariate data. The results can be found below.

Best constellations of "Rocket" and "Minirocket" (the detailed tables are also included further below):

Method  | Lag | Num. of kernels | MAE
------------- |-----|------| -------------
Rocket-avg | 10  | 10000   | 64.9
Rocket-avg+std | 10  | 5000    | 65.0
Rocket-avg+std,min,max | 10  | 100   | 72.6
Rocket-ppv-max | 10  | 10000 | 78.2
Minirocket | 10  | 10000 | 78.2

### Comparison of MAE of different methods:

Method  | MAE
------------- | -------------
Rocket-ppv-max | 78.2
Rocket-avg | 64.9
Rocket-avg+std | 65.0
Rocket-avg+std,min,max | 72.6
Minirocket(Lag 10, Kernel 1000) | 1874.5
Multirocket(Lag 10) | 802.8
Multirocket(Lag 20) | 1055.9
InceptionTime(Lag 10) | 426.3
InceptionTime(Lag 20) | 229

Poor performance of Minirocket and Multirocket can be explained by their almost deterministic nature.

### Detailed table view of MAE of various set-up of Rocket and Minirocket
**1. Rocket-avg**

 Lag          | Num. of kernels | MAE
--------------|----------------| -------------
 10           | 10             | 73.7
 10           | 100            | 74.3
 10           | 1000           | 68.6
 10           | 5000           | 67.6
 10           | 10000          | 64.9
 20           | 10             | 93.6
 20           | 100            | 68.7
 20           | 1000           | 66.0
 20           | 5000           | 77.8
 20           | 10000          | 71.7

**2. Rocket-avg+std**

Lag          | Num. of kernels | MAE
--------------|-----------------| -------------
10           | 10              | 89.5
10           | 100             | 74.8
10           | 1000            | 72.5
10           | 5000            | 65.0
10           | 10000           | 108.8
20           | 10              | 85.5
20           | 100             | 82.1
20           | 1000            | 81.3
20           | 5000            | 70.5
20           | 10000           | 99.6

**3. Rocket-avg+std,min,max**

Lag          | Num. of kernels | MAE
--------------|-----------------| -------------
10           | 10              | 88.5
10           | 100             | 72.6
10           | 1000            | 84.2
10           | 5000            | 96.3
10           | 10000           | 137.2
20           | 10              | 95.8
20           | 100             | 114.0
20           | 1000            | 83.4
20           | 5000            | 21686.2
20           | 10000           | 735.3


**4. Rocket-ppv-max**

Lag          | Num. of kernels | MAE
--------------|-----------------| -------------
10           | 10              | 79.9
10           | 100             | 92.1
10           | 1000            | 134.6
10           | 5000            | 102.4
10           | 10000           | 78.2
20           | 10              | 116.9
20           | 100             | 112.4
20           | 1000            | 110.5
20           | 5000            | 573.1
20           | 10000           | 121.6

**4. Minirocket**

Lag          | Num. of kernels | MAE
--------------|----------------| -------------
10           | 100            | 1933.6
10           | 1000           | 1874.5
10           | 5000           | 2063.0
10           | 10000          | 2211.2
20           | 100            | 2558.8
20           | 1000           | 2441.2
20           | 5000           | 4314.8
20           | 10000          | 3039.8