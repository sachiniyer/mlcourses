.PHONY: lib, pybind, clean, format, all

all: lib


lib:
	@mkdir -p build
	@cd build; cmake -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_C_COMPILER=/usr/bin/gcc ..
	@cd build; $(MAKE)

format:
	python3 -m black .
	clang-format -i src/*.cc src/*.cu

clean:
	rm -rf build python/needle/backend_ndarray/ndarray_backend*.so
