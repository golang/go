# Go Tools

This subrepository holds the source for various packages and tools that support
the Go programming language.

Some of the tools, `godoc` and `vet` for example, are included in binary Go
distributions.

Others, including the Go `guru` and the test coverage tool, can be fetched with
`go get`.

Packages include a type-checker for Go and an implementation of the
Static Single Assignment form (SSA) representation for Go programs.

## Download/Install

The easiest way to install is to run `go get -u golang.org/x/tools/...`. You can
also manually git clone the repository to `$GOPATH/src/golang.org/x/tools`.

## Report Issues / Send Patches

This repository uses Gerrit for code changes. To learn how to submit changes to
this repository, see https://golang.org/doc/contribute.html.

The main issue tracker for the tools repository is located at
https://github.com/golang/go/issues. Prefix your issue with "x/tools/(your
subdir):" in the subject line, so it is easy to find.
