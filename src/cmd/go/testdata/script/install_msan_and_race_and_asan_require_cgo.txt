# Tests Issue #21895

env CGO_ENABLED=0

[GOOS:darwin] [!short] [race] go build -race triv.go

[!GOOS:darwin] [race] ! go install -race triv.go
[!GOOS:darwin] [race] stderr '-race requires cgo'
[!GOOS:darwin] [race] ! stderr '-msan'

[msan] ! go install -msan triv.go
[msan] stderr '-msan requires cgo'
[msan] ! stderr '-race'

[asan] ! go install -asan triv.go
[asan] stderr '(-asan: the version of $(go env CC) could not be parsed)|(-asan: C compiler is not gcc or clang)|(-asan is not supported with [A-Za-z]+ compiler (\d+)\.(\d+))|(-asan requires cgo)'
[asan] ! stderr '-msan'

-- triv.go --
package main

func main() {}
