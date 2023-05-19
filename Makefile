## UmbrellaIntegrator Makefile

# compiler and flags
F2PY = python3 -m numpy.f2py --fcompiler="gnu95" --opt="-O3" --f90flags="-fopenmp" -lgomp

all:
	$(MAKE) clean
	$(F2PY) -c umbrellaint_fortran.f90 -m umbrellaint_fortran

clean:
	rm -f *.so
