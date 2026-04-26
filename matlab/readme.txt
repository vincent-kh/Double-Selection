Information for each folder

(1) data: all required data to obtain the results

(2) functions: functions for our method in the paper, and other necessary functions to obtain the results

(3) glmnet_matlab: the glmnet package for Matlab used in our function

(4) main: to produce the main results

(5) robustness: produce robustness table

(6) simulation: to produce an example for the simulation results (tables + figures)

For speed, we implement our procedure using the glmnet package <http://web.stanford.edu/~hastie/glmnet_matlab/> on Matlab 2014b. You may have incompatibility issue for later versions of Matlab. A simple workaround might be to place a file named "libgfortran.3.dylib" (not included) to the designated fortran library, e.g., “/usr/local/gfortran/lib/” on a Mac.