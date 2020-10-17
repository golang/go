## steps to compile go
roughly followed this guide
https://github.com/golang/go/wiki/WindowsBuild
and ran the executable

install MinGW latest
GCC for Windows 64 & 32 bits
http://mingw-w64.org/doku.php

https://jmeubank.github.io/tdm-gcc/

## Go testing
go test -coverprofile=coverage.out ./...
go tool cover -func=coverage.out
go tool cover -html=coverage.out

## encoding/xml notes

### detailed test coverage
go test -coverprofile=coverage.out
go tool cover -func=coverage.out
go tool cover -html=coverage.out

go test -coverprofile=coverage.out && go tool cover -html=coverage.out

### top-level coverage
go test

### run specific test
go test -v -run TestDecodeEOF

### TODO
0. modify animal custom marshaller example and add namespaces
   (see if test coverage luckily goes up)
1. marshal_test.go
2. add logs and look for close-by corner cases in the big loop
3. bring coverage up to 95%

reach 95% code coverage
ticket creator asked for
> more tests for name spaces, custom marshalers, custom
unmarshalers, and the interaction between all those

## Links
https://go-review.googlesource.com/dashboard/self
https://golang.org/doc/contribute.html
https://github.com/golang/go/issues/6094
https://golang.org/pkg/encoding/xml/#example_Encoder