# User guide

**If you're having issues with `gopls`, please see the
[troubleshooting guide](troubleshooting.md).**

## Editors

The following is the list of editors with known integrations for `gopls`.

* [VSCode](vscode.md)
* [Vim / Neovim](vim.md)
* [Emacs](emacs.md)
* [Acme](acme.md)
* [Sublime Text](subl.md)
* [Atom](atom.md)

If you use `gopls` with an editor that is not on this list, please let us know
by [filing an issue](#new-issue) or [modifying this documentation](contributing.md).

## Overview

* [Installation](#installation)
* [Configuration](#configuration)

Learn more at the following pages:

* [Features](features.md)
* [Command-line](command-line.md)

## Installation

For the most part, you should not need to install or update `gopls`. Your editor should handle that step for you.

If you do want to get the latest stable version of `gopls`, change to any directory that is both outside of your `GOPATH` and outside of a module (a temp directory is fine), and run

```sh
go get golang.org/x/tools/gopls@latest
```

**Do not** use the `-u` flag, as it will update your dependencies to incompatible versions.

To get a specific version of `gopls` (for example, to test a prerelease
version), run:

```sh
go get golang.org/x/tools/gopls@vX.Y.Z
```

Where `vX.Y.Z` is the desired version.

If you see this error:

```sh
$ go get golang.org/x/tools/gopls@latest
go: cannot use path@version syntax in GOPATH mode
```

then run

```sh
GO111MODULE=on go get golang.org/x/tools/gopls@latest
```

### Unstable versions

`go get` doesn't honor the `replace` directive in the `go.mod` of
`gopls` when you are outside of the `gopls` module, so a simple `go get`
with `@master` could fail.  To actually update your `gopls` to the
latest **unstable** version, use:

```sh
go get golang.org/x/tools/gopls@master golang.org/x/tools@master
```

In general, you should use `@latest` instead, to prevent frequent
breakages.

### Supported Go versions

`gopls` follows the
[Go Release Policy](https://golang.org/doc/devel/release.html#policy),
meaning that it officially supports the last 2 major Go releases. We run CI to
verify that the `gopls` tests pass for the last 4 major Go releases, but do not
prioritize issues only affecting legacy Go release (3 or 4 releases ago).

## Configuration

### Environment variables

These are often inherited from the editor that launches `gopls`, and sometimes
the editor has a way to add or replace values before launching. For example,
VSCode allows you to configure `go.toolsEnvVars`.

Configuring your environment correctly is important, as `gopls` relies on the
`go` command.

### Command-line flags

See the [command-line page](command-line.md) for more information about the
flags you might specify. All editors support some way of adding flags to
`gopls`, for the most part you should not need to do this unless you have very
unusual requirements or are trying to [troubleshoot](troubleshooting.md#steps)
`gopls` behavior.

### Editor settings

For the most part these will be settings that control how the editor interacts
with or uses the results of `gopls`, not modifications to `gopls` itself. This
means they are not standardized across editors, and you will have to look at
the specific instructions for your editor integration to change them.

#### The set of workspace folders

This is one of the most important pieces of configuration. It is the set of
folders that gopls considers to be "roots" that it should consider files to
be a part of.

If you are using modules there should be one of these per go.mod that you
are working on. If you do not open the right folders, very little will work.
**This is the most common misconfiguration of `gopls` that we see**.

#### Global configuration

There should be a way of declaring global settings for `gopls` inside the
editor. The settings block will be called `"gopls"` and contains a collection
of controls for `gopls` that the editor is not expected to understand or
control.

In VSCode, this would be a section in your settings file that might look like
this:

```json5
  "gopls": {
    "usePlaceholders": true,
    "completeUnimported": true
  },
```

See [Settings](settings.md) for more information about the available
configurations.

#### Workspace folder configuration

This contains exactly the same set of values that are in the global
configuration, but it is fetched for every workspace folder separately.
The editor can choose to respond with different values per-folder.

### Working on the Go source distribution

If you are working on the [Go project](https://go.googlesource.com/go) itself,
your `go` command will have to correspond to the version of the source you are
working on. That is, if you have downloaded the code to `$HOME/go`, your `go`
command should be the `$HOME/go/bin/go` executable that you built with
`make.bash` or equivalent.

You can achieve this by adding the right version of `go` to your `PATH` (`export PATH=$HOME/go/bin:$PATH` on Unix systems) or by configuring your editor. In VS Code, you can use the `go.alternateTools` setting to point to the correct version of `go`:

```json5
{

    "go.alternateTools": {
        "go": "$HOME/bin/go"
    }
}
```
