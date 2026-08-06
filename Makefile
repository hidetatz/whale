test-short:
	WHALE_BACKEND=PYTHON python test.py

test: test-short
	WHALE_BACKEND=CLANG python test.py
	WHALE_BACKEND=CUDA python test.py
