export LD_LIBRARY_PATH += /opt/OpenBLAS/lib

format:
	goimports -w .

test:
	go test -short ./...

testv:
	go test -v -short ./...

testall:
	go test ./...

testallv:
	go test -v ./...
