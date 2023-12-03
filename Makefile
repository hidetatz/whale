format:
	goimports -w .

test:
	go test ./...

testv:
	go test -v ./...
