## Tools {#tools}

<!-- go.dev/issue/77177 -->

Response file (`@file`) parsing is now supported for the `compile`, `link`, `asm`, `cgo`, `cover`, and `pack` tools.
The response file contains whitespace-separated arguments with support for single-quoted and double-quoted strings, escape sequences, and backslash-newline line continuation.
The format is compatible with GCC's response file implementation to ensure interoperability with existing build systems.

### Go command {#go-command}

`go test` now invokes the stdversion vet check by default.
This reports the use of standard library symbols that are too new
for the Go version in force in the referring file,
as determined by `go` directive in `go.mod` and build tags on the file.

<!-- go.dev/issue/78090 -->

The `go` command no longer has support for the bzr version control system.
It will no longer be able to directly fetch modules hosted on bzr servers.

### Cgo {#cgo}

