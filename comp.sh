rm -rf build
rm fft_cython.so
rm fft_cython_py.c
python setup.py build_ext --inplace
