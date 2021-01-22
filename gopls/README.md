# `gopls`, the Go language server

[![PkgGoDev](https://pkg.go.dev/badge/golang.org/x/tools/gopls)](https://pkg.go.dev/golang.org/x/tools/gopls)

`gopls` (pronounced "Go please") is the official Go [language server] developed
by the Go team. It provides IDE features to any [LSP]-compatible editor.

<!--TODO(rstambler): Add gifs here.-->

You should not need to interact with `gopls` directly--it will be automatically
integrated into your editor. The specific features and settings vary slightly
by editor, so we recommend that you proceed to the [documentation for your
editor](#editors) below.

## Editors

To get started with `gopls`, install an LSP plugin in your editor of choice.

* [VSCode](https://github.com/golang/vscode-go/blob/master/README.md)
* [Vim / Neovim](doc/vim.md)
* [Emacs](doc/emacs.md)
* [Atom](https://github.com/MordFustang21/ide-gopls)
* [Sublime Text](doc/subl.md)
* [Acme](https://github.com/fhs/acme-lsp)

If you use `gopls` with an editor that is not on this list, please let us know
by [filing an issue](#new-issue) or [modifying this documentation](doc/contributing.md).

## Installation

For the most part, you should not need to install or update `gopls`. Your
editor should handle that step for you.

If you do want to get the latest stable version of `gopls`, change to any
directory that is both outside of your `GOPATH` and outside of a module (a temp
directory is fine), and run:

```sh
GO111MODULE=on go get golang.org/x/tools/gopls@latest
```

**NOTE**: Do not use the `-u` flag, as it will update your dependencies to
incompatible versions.

Learn more in the [advanced installation
instructions](doc/advanced.md#installing-unreleased-versions).

## Setting up your workspace

`gopls` supports both Go module and GOPATH modes, but if you are working with
multiple modules or uncommon project layouts, you will need to specifically
configure your workspace. See the [Workspace document](doc/workspace.md) for
information on supported workspace layouts.

## Configuration

You can configure `gopls` to change your editor experience or view additional
debugging information. Configuration options will be made available by your
editor, so see your [editor's instructions](#editors) for specific details. A
full list of `gopls` settings can be found in the [Settings documentation](doc/settings.md).

### Environment variables

`gopls` inherits your editor's environment, so be aware of any environment
variables you configure. Some editors, such as VS Code, allow users to
selectively override the values of some environment variables.

## Troubleshooting

If you are having issues with `gopls`, please follow the steps described in the
[troubleshooting guide](doc/troubleshooting.md).

## Supported Go versions and build systems

`gopls` follows the
[Go Release Policy](https://golang.org/doc/devel/release.html#policy),
meaning that it officially supports the last 2 major Go releases. Though we
try not to break older versions, we do not prioritize issues only affecting
legacy Go releases.

`gopls` currently only supports the `go` command, so if you are using a
different build system, `gopls` will not work well. Bazel support is currently
blocked on
[bazelbuild/rules_go#512](https://github.com/bazelbuild/rules_go/issues/512).

## Additional information

* [Features](doc/features.md)
* [Command-line interface](doc/command-line.md)
* [Advanced topics](doc/advanced.md)
* [Contributing to `gopls`](doc/contributing.md)
* [Integrating `gopls` with an editor](doc/design/integrating.md)
* [Design requirements and decisions](doc/design/design.md)
* [Implementation details](doc/design/implementation.md)
* [Open issues](https://github.com/golang/go/issues?q=is%3Aissue+is%3Aopen+label%3Agopls)

[language server]: https://langserver.org
[LSP]: https://microsoft.github.io/language-server-protocol/
[Gophers Slack]: https://gophers.slack.com/
