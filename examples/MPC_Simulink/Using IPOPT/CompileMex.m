casadi.GlobalOptions.getCasadiPath();
inc_path = casadi.GlobalOptions.getCasadiIncludePath();
mex('-v',['-I' inc_path],['-L' lib_path],'-lcasadi', 'casadi_fun.c')