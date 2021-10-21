# Advanced topics

This documentation is for advanced `gopls` users, who may want to test
unreleased versions or try out special features.

## Installing unreleased versions

To get a specific version of `gopls` (for example, to test a prerelease
version), run:

```sh
GO111MODULE=on go install golang.org/x/tools/gopls@vX.Y.Z
```

Where `vX.Y.Z` is the desired version.

### Unstable versions

To update `gopls` to the latest **unstable** version, use:

```sh
# Create an empty go.mod file, only for tracking requirements.
cd $(mktemp -d)
go mod init gopls-unstable

# Use 'go get' to add requirements and to ensure they work together.
go get golang.org/x/tools/gopls@master golang.org/x/tools@master

# For go1.17 or older, the above `go get` command will build and
# install `gopls`. For go1.18+ or tip, run the following command to install
# using selected versions in go.mod.
go install golang.org/x/tools/gopls
```

## Working on the Go source distribution

If you are working on the [Go project] itself, the `go` command that `gopls`
invokes will have to correspond to the version of the source you are working
on. That is, if you have checked out the Go project to `$HOME/go`, your `go`
command should be the `$HOME/go/bin/go` executable that you built with
`make.bash` or equivalent.

You can achieve this by adding the right version of `go` to your `PATH`
(`export PATH=$HOME/go/bin:$PATH` on Unix systems) or by configuring your
editor.

## Working with generic code

Gopls has experimental support for generic Go, as defined by the type
parameters proposal ([golang/go#43651](https://golang.org/issues/43651)) and
type set addendum ([golang/go#45346](https://golang.org/issues/45346)).

To enable this support, you need to build gopls with a version of Go that
supports type parameters, currently just tip. This can be done by checking
out the `master` branch in the Go repository, or by using
`golang.org/dl/gotip`:

```
$ go install golang.org/dl/gotip@latest
$ gotip download
```

For building gopls with type parameter support, it is recommended that you
build gopls at tip. External APIs are under active development on the Go
`master` branch, so building gopls at tip minimizes the chances of a build
failure. You will also need to update the `go` directive in your `go.mod`
file to refer to `go 1.18` to use generics.

Build and install the latest **unstable** version of `gopls` following
[the instruction](#installing-unreleased-versions).
Remember to use `gotip` instead of `go`.

The `gopls` build with this instruction understands generic code. To actually
run the generic code you develop, you must also use the tip version of the Go
compiler. For example:

```
$ gotip run .
```

[Go project]: https://go.googlesource.com/go
