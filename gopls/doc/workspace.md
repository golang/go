# Setting up your workspace

`gopls` supports both Go module and GOPATH modes. However, it needs a defined
scope in which language features like references, rename, and implementation
should operate.

The following options are available for configuring this scope:

## Module mode

### One module

If you are working with a single module, you can open the module root (the
directory containing the `go.mod` file), a subdirectory within the module,
or a parent directory containing the module.

**Note**: If you open a parent directory containing a module, it must **only**
contain that single module. Otherwise, you are working with multiple modules.

### Multiple modules

Gopls has several alternatives for working on multiple modules simultaneously,
described below. Starting with Go 1.18, Go workspaces are the preferred solution.

#### Go workspaces (Go 1.18+)

Starting with Go 1.18, the `go` command has native support for multi-module
workspaces, via [`go.work`](https://go.dev/ref/mod#workspaces) files. These
files are recognized by gopls starting with `gopls@v0.8.0`.

The easiest way to work on multiple modules in Go 1.18 and later is therefore
to create a `go.work` file containing the modules you wish to work on, and set
your workspace root to the directory containing the `go.work` file.

For example, suppose this repo is checked out into the `$WORK/tools` directory.
We can work on both `golang.org/x/tools` and `golang.org/x/tools/gopls`
simultaneously by creating a `go.work` file:

```
cd $WORK
go work init
go work use tools tools/gopls
```

...followed by opening the `$WORK` directory in our editor.

#### Experimental workspace module (Go 1.17 and earlier)

With earlier versions of Go, `gopls` can simulate multi-module workspaces by
creating a synthetic module requiring the the modules in the workspace root.
See [the design document](https://github.com/golang/proposal/blob/master/design/37720-gopls-workspaces.md)
for more information.

This feature is experimental, and will eventually be removed once `go.work`
files are accepted by all supported Go versions.

You can enable this feature by configuring the
[experimentalWorkspaceModule](settings.md#experimentalworkspacemodule-bool)
setting.

#### Multiple workspace folders

If neither of the above solutions work, and your editor allows configuring the
set of
["workspace folders"](https://microsoft.github.io/language-server-protocol/specifications/specification-3-17/#workspaceFolder)
used during your LSP session, you can still work on multiple modules by adding
a workspace folder at each module root (the locations of `go.mod` files). This
means that each module has its own scope, and features will not work across
modules. 

In VS Code, you can create a workspace folder by setting up a
[multi-root workspace](https://code.visualstudio.com/docs/editor/multi-root-workspaces).
View the [documentation for your editor plugin](../README.md#editor) to learn how to
configure a workspace folder in your editor.

### GOPATH mode

When opening a directory within your GOPATH, the workspace scope will be just
that directory.

### At your own risk

Some users or companies may have projects that encompass one `$GOPATH`. If you
open your entire `$GOPATH` or `$GOPATH/src` folder, the workspace scope will be
your entire `GOPATH`. If your GOPATH is large, `gopls` to be very slow to start
because it will try to find all of the Go files in the directory you have
opened. It will then load all of the files it has found.

To work around this case, you can create a new `$GOPATH` that contains only the
packages you want to work on.

---

If you have additional use cases that are not mentioned above, please
[file a new issue](https://github.com/golang/go/issues/new).
