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

As of Jan 2021, if you are working with multiple modules or nested modules, you
will need to create a "workspace folder" for each module. This means that each
module has its own scope, and features will not work across modules. We are
currently working on addressing this limitation--see details about
[experimental workspace module mode](#workspace-module-experimental)
below.

In VS Code, you can create a workspace folder by setting up a
[multi-root workspace](https://code.visualstudio.com/docs/editor/multi-root-workspaces).
View the [documentation for your editor plugin](../README.md#editor) to learn how to
configure a workspace folder in your editor.

#### Workspace module (experimental)

Many `gopls` users would like to work with multiple modules at the same time
([golang/go#32394](https://github.com/golang/go/issues/32394)), and
specifically, have features that work across modules. We plan to add support
for this via a concept called the "workspace module", which is described in
[this design document](https://github.com/golang/proposal/blob/master/design/37720-gopls-workspaces.md).
This feature works by creating a temporary module that requires all of your
workspace modules, meaning all of their dependencies must be compatible.

The workspace module feature is currently available as an opt-in experiment,
and it will allow you to work with multiple modules without creating workspace
folders for each module. You can try it out by configuring the
[experimentalWorkspaceModule](settings.md#experimentalworkspacemodule-bool)
setting. If you try it and encounter issues, please
[report them](https://github.com/golang/go/issues/new) so we can address them
before the feature is enabled by default.

You can follow our progress on the workspace module work by looking at the
open issues in the
[gopls/workspace-module milestone](https://github.com/golang/go/milestone/179).

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
