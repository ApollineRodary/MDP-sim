swig -c++ -python swig/pymdp.i
python3 -W ignore swig/setup.py build_ext --build-lib=swig
rm swig/pymdp_wrap.cxx