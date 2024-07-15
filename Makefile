export LD_LIBRARY_PATH += /opt/OpenBLAS/lib

format:
	goimports -w .

test:
	go test -short ./...

testv:
	go test -v -short ./...

testall:
	LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:/opt/OpenBLAS/lib go test ./...

testallv:
	LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:/opt/OpenBLAS/lib go test -v ./...
