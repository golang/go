# User guide

##### If you're having issues with `gopls`, please see the [troubleshooting guide](troubleshooting.md).

This document focuses on VSCode, as at the time of writing, VSCode is the most popular Go editor. However, most of the features described here work in any editor. The settings should be easy to translate to those of another editor's LSP client. The differences will be in the place where you define the settings and the syntax with which you declare them.

## Editors

The following is the list of editors with known integrations.
If you use `gopls` with an editor that is not on this list, please let us know by [filing an issue](#new-issue) or [modifying this documentation](#contribute).

* [VSCode](vscode.md)
* [Vim / Neovim](vim.md)
* [Emacs](emacs.md)
* [Acme](acme.md)
* [Sublime Text](subl.md)
* [Atom](atom.md)

## Installation

For the most part, you should not need to install or update `gopls`. Your editor should handle that step for you.

If you do want to get the latest stable version of `gopls`, change to any directory that is both outside of your `GOPATH` and outside of a module (a temp directory is fine), and run

```sh
go get golang.org/x/tools/gopls@latest
```

**Do not** use the `-u` flag, as it will update your dependencies to incompatible versions.

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
$ go get golang.org/x/tools/gopls@master golang.org/x/tools@master
```

In general, you should use `@latest` instead, to prevent frequent
breakages.

## Configurations

### Environment variables

These are often inherited from the editor that launches `gopls`, and sometimes the editor has a way to add or replace values before launching. For example, VSCode allows you to configure `go.toolsEnvVars`.

Configuring your environment correctly is important, as `gopls` relies on the `go` command.

### Command line flags

See the [command line page](command-line.md) for more information about the flags you might specify.
All editors support some way of adding flags to `gopls`, for the most part you should not need to do this unless you have very unusual requirements or are trying to [troubleshoot](troubleshooting.md#steps) `gopls` behavior.

### Editor settings

For the most part these will be settings that control how the editor interacts with or uses the results of `gopls`, not modifications to `gopls` itself. This means they are not standardized across editors, and you will have to look at the specific instructions for your editor integration to change them.

#### The set of workspace folders

This is one of the most important pieces of configuration. It is the set of folders that gopls considers to be "roots" that it should consider files to be a part of.

If you are using modules there should be one of these per go.mod that you are working on.
If you do not open the right folders, very little will work. **This is the most common misconfiguration of `gopls` that we see**.

#### Global configuration

There should be a way of declaring global settings for `gopls` inside the editor. The settings block will be called `"gopls"` and contains a collection of controls for `gopls` that the editor is not expected to understand or control.

In VSCode, this would be a section in your settings file that might look like this:

```json5
  "gopls": {
    "usePlaceholders": true,
    "completeUnimported": true
  },
```

See [Settings](settings.md) for more information about the available configurations.

#### Workspace folder configuration

This contains exactly the same set of values that are in the global configuration, but it is fetched for every workspace folder separately. The editor can choose to respond with different values per-folder.

## Command line support

Much of the functionality of `gopls` is available through a command line interface.

There are two main reasons for this. The first is that we do not want users to rely on separate command line tools when they wish to do some task outside of an editor. The second is that the CLI assists in debugging. It is easier to reproduce behavior via single command.

It is not a goal of `gopls` to be a high performance command line tool. Its command line is intended for single file/package user interaction speeds, not bulk processing.

For more information, see the `gopls` [command line page](command-line.md).
