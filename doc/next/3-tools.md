## Tools {#tools}

### Go command {#go-command}

`go test` now invokes the stdversion vet check by default.
This reports the use of standard library symbols that are too new
for the Go version in force in the referring file,
as determined by `go` directive in `go.mod` and build tags on the file.

### Cgo {#cgo}

