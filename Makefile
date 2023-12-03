format:
	goimports -w .

test:
	go test ./*.go

testv:
	go test -v ./*.go
