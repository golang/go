# Advanced topics

This documentation is for advanced `gopls` users, who may want to test
unreleased versions or try out special features.

## Installing unreleased versions

To get a specific version of `gopls` (for example, to test a prerelease
version), run:

```sh
GO111MODULE=on go get golang.org/x/tools/gopls@vX.Y.Z
```

Where `vX.Y.Z` is the desired version.

### Unstable versions

To update `gopls` to the latest **unstable** version, use:

```sh
GO111MODULE=on go get golang.org/x/tools/gopls@master golang.org/x/tools@master
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

[Go project]: https://go.googlesource.com/go
