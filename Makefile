build:
	./build.sh
test:
	PYTHONPATH=./src pytest
lint:
	pylint --disable=E0401,E1101,W0201,W0613,W0719,C0202,C0301,R0801,R0902 src/
dist/archive.tar.gz:
	tar -czvf dist/archive.tar.gz dist/main