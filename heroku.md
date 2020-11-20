# Files Required by Heroku
- setup.sh
- Procfile
- runtime.txt

# Heroku Limitations
- The total file size must be less than 500 MB.

```bash
# python module file sizes
tensorflow   320.4 MB
scipy        25.9 MB
pyarrow      17.7 MB
numpy        14.5 MB
matplotlib   11.6 MB
pandas       9.5 MB
streamlit    7.4 MB
scikit-learn 6.8 MB

Total size was 535.5MB.
I deleted the processed file of size 42.8MB. Still the slug failed.
This time total size is 518.3M
```

# Tensorflow size
The tensorflow occupies large space. So I looked into its module directory
and found following files larger than 2 MB. All the larger files are
shared objects (`.so`) or dynamic libraries (`.dylib`) or header files (`.h`).
I can not delete these files. Tensorflow is not so great library to deploy
the models in web.
```
$ larger 2
./python/_pywrap_tensorflow_internal.so: 398M
./python/_pywrap_tfcompile.so          : 39M
./libtensorflow_framework.2.3.0.dylib  : 28M
./libtensorflow_framework.2.dylib      : 28M
./libtensorflow_framework.dylib        : 28M

./python/profiler/internal/_pywrap_profiler.so                             : 8.4M
./lite/python/optimize/_pywrap_tensorflow_lite_calibration_wrapper.so      : 4.6M
./lite/python/interpreter_wrapper/_pywrap_tensorflow_interpreter_wrapper.so: 4.4M
./include/external/sobol_data/sobol_data.h                                 : 4.4M
./python/_pywrap_tf_item.so                                                : 2.7M
```