#
#@BEGIN LICENSE
#
# gpu_dfcc by Eugene DePrince, a plugin to:
#
# PSI4: an ab initio quantum chemistry software package
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
#@END LICENSE
#

PSI4SRC = /shares/rogers/psi4public
PSI4 = /shares/rogers/psi4
# nvidia compiler
NVCC=nvcc
CUDALIBS = -lcublas -L/apps/cuda/6.5.14/lib64
CUDAFLAGS = -arch sm_13 -Xcompiler -fopenmp --compiler-options '-fPIC' -O2


#
# Plugin Makefile generated by Psi4.
#
# You shouldn't need to modify anything below this line
#

# The name of your plugin. Taken from the directory name.
NAME = $(shell basename `pwd`)

# C++ source files for your plugin. By default we grab all *.cc files.
CXXSRC  = $(notdir $(wildcard *.cc))
CUDASRC = $(notdir $(wildcard *.cu))

# Flags that were used to compile Psi4.
CXX = icpc
CXXDEFS = -DFC_SYMBOL=2 -DHAVE_MKL_LAPACK -DHAVE_MKL_BLAS -DHAS_CXX11_VARIADIC_TEMPLATES -DHAS_CXX11_STATIC_ASSERT -DHAS_CXX11_SIZEOF_MEMBER -DHAS_CXX11_RVALUE_REFERENCES -DHAS_CXX11_NULLPTR -DHAS_CXX11_LONG_LONG -DHAS_CXX11_LAMBDA -DHAS_CXX11_INITIALIZER_LIST -DHAS_CXX11_DECLTYPE -DHAS_CXX11_CSTDINT_H -DHAS_CXX11_CONSTEXPR -DHAS_CXX11_AUTO_RET_TYPE -DHAS_CXX11_AUTO -DHAS_CXX11_FUNC -DHAS_CXX11 -DSYS_LINUX
CXXFLAGS = -DRESTRICT=__restrict__ -Xlinker -export-dynamic -fPIC -std=c++11 -mkl=parallel -openmp -O3 -no-prec-div -DNDEBUG -xHost -ggdb
INCLUDES = -I$(PSI4)/src/lib -I$(PSI4SRC)/src/lib -I$(PSI4SRC)/include -I$(PSI4)/include -I$(PSI4)/boost/include -I/usr/include/python2.6 -I/usr/include -I/usr/include -I/usr/include -I/usr/include
OBJDIR = $(PSI4)

# Used to determine linking flags.
UNAME = $(shell uname)

# Need to link against Psi4 plugin library
PSIPLUGIN = -L$(OBJDIR)/lib -lplugin

DEPENDINCLUDE = $(notdir $(wildcard *.h*))

PSITARGET = $(NAME).so

# Start the compilation rules
default:: $(PSITARGET)

# Cuda libraries:
NVCCINCLUDE=-I$(top_srcdir)/include -I$(OBJDIR)/include -I$(OBJDIR)/src/lib $(INCLUDES)

# Add the flags needed for shared library creation
ifeq ($(UNAME), Linux)
    LDFLAGS = -shared $(CUDALIBS)
endif
ifeq ($(UNAME), Darwin)
    LDFLAGS = -shared -undefined dynamic_lookup $(CUDALIBS)
    CXXFLAGS += -fno-common
endif

# The object files
BINOBJ  = $(CXXSRC:%.cc=%.o)
CUDAOBJ = $(CUDASRC:%.cu=%.o)

%.o: %.cc
	$(CXX) $(CXXDEFS) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(CUDAFLAGS) $(NVCCINCLUDE) -c $< $(OUTPUT_OPTION)

$(PSITARGET): $(BINOBJ) $(CUDAOBJ)
	$(NVCC) $(LDFLAGS) -o $@ $^ $(CXXDBG) $(PSILIBS)

# Erase all compiled intermediate files
clean:
	rm -f $(CUDAOBJ) $(BINOBJ) $(PSITARGET) *.d *.pyc *.test output.dat psi.timer.dat

