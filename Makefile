# A stripped down makefile from xgboost
# base compile flags
CFLAGS += -Ofast -O3

# open MP flags (enable by default)
USE_OPENMP ?= 1
ifeq ($(USE_OPENMP), 1)
	OPENMP_FLAGS = -fopenmp
else
	OPENMP_FLAGS = -DDISABLE_OPENMP
endif

export CFLAGS += $(OPENMP_FLAGS)

# set compiler defaults for OSX versus *nix
OS := $(shell uname)
ifeq ($(OS), Darwin)
	ifeq ($(USE_OPENMP), 1)
		export CC ?= gcc-7
	endif
	export CC ?= $(if $(shell which clang), clang, gcc)

	ifeq ($(USE_OPENMP), 1)
		export CXX ?= g++-7
	endif
	export CXX ?= $(if $(shell which clang++), clang++, g++)

else
# linux defaults
	export CC ?= gcc
	export CXX ?= g++
endif


# build the library
all: clean install

build:
	python setup.py build_ext --inplace --force

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
