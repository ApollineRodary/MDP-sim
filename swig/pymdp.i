%module pymdp

%{
#define SWIG_FILE_WITH_INIT
#include "../src/mdp.hpp"
%}

%include "../src/mdp.hpp"