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

To update `gopls` to the latest **unstable** version, use the following
commands.

```sh
# Create an empty go.mod file, only for tracking requirements.
cd $(mktemp -d)
go mod init gopls-unstable

# Use 'go get' to add requirements and to ensure they work together.
go get -d golang.org/x/tools/gopls@master golang.org/x/tools@master

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

Gopls has support for editing generic Go code. To enable this support, you need
to **install gopls using Go 1.18 or later**. The easiest way to do this is by
[installing Go 1.18+](https://go.dev/dl) and then using this Go version to
install gopls:

```
$ go install golang.org/x/tools/gopls@latest
```

It is strongly recommended that you install the latest version of `gopls`, or
the latest **unstable** version as [described above](#installing-unreleased-versions).
We're still working on improving our generics support.

The `gopls` built with these instructions understands generic code. See the
[generics tutorial](https://go.dev/doc/tutorial/generics) for more information
on how to use generics in Go!

### Known issues

  * [`staticcheck`](https://github.com/golang/tools/blob/master/gopls/doc/settings.md#staticcheck-bool)
    on generic code is not supported yet.

[Go project]: https://go.googlesource.com/go
