# A stripped down makefile from xgboost
ifndef USE_OPENMP
	USE_OPENMP = 1
endif

# base compile flags
CFLAGS_ =

# open MP flags
OPENMP_FLAGS =
ifeq ($(USE_OPENMP), 1)
	OPENMP_FLAGS = -fopenmp
else
	OPENMP_FLAGS = -DDISABLE_OPENMP
endif

CFLAGS_ += $(OPENMP_FLAGS)

# set compiler defaults for OSX versus *nix
OS := $(shell uname)
ifeq ($(OS), Darwin)
	ifeq ($(USE_OPENMP), 1)
		export CC = gcc-7
	endif
	ifeq ($(USE_OPENMP), 1)
		export CXX = g++-7
	endif
	ifndef CC
		export CC = $(if $(shell which clang), clang, gcc)
	endif
	ifndef CXX
		export CXX = $(if $(shell which clang++), clang++, g++)
	endif
else
# linux defaults
	ifndef CC
		export CC = gcc
	endif
	ifndef CXX
		export CXX = g++
	endif
endif

export CFLAGS = $(CFLAGS_)

# build the library
all: install

build:
	python setup.py build_ext --inplace

install: build
	python setup.py install

clean: clean_build clean_install

clean_build:
	$(RM) -rf build
	$(RM) -rf sgimc/*.so
	$(RM) -rf sgimc/ops.c

clean_install:
	$(RM) -rf dist
	$(RM) -rf sgimc.egg-info
