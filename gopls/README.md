# `gopls`, the Go language server

[![PkgGoDev](https://pkg.go.dev/badge/golang.org/x/tools/gopls)](https://pkg.go.dev/golang.org/x/tools/gopls)

`gopls` (pronounced "Go please") is the official Go [language server] developed
by the Go team. It provides IDE features to any [LSP]-compatible editor.

<!--TODO(rfindley): Add gifs here.-->

You should not need to interact with `gopls` directly--it will be automatically
integrated into your editor. The specific features and settings vary slightly
by editor, so we recommend that you proceed to the
[documentation for your editor](#editors) below.

## Editors

To get started with `gopls`, install an LSP plugin in your editor of choice.

* [VS Code](https://github.com/golang/vscode-go/blob/master/README.md)
* [Vim / Neovim](doc/vim.md)
* [Emacs](doc/emacs.md)
* [Atom](https://github.com/MordFustang21/ide-gopls)
* [Sublime Text](doc/subl.md)
* [Acme](https://github.com/fhs/acme-lsp)
* [Lapce](https://github.com/lapce-community/lapce-go)

If you use `gopls` with an editor that is not on this list, please send us a CL
[updating this documentation](doc/contributing.md).

## Installation

For the most part, you should not need to install or update `gopls`. Your
editor should handle that step for you.

If you do want to get the latest stable version of `gopls`, run the following
command:

```sh
go install golang.org/x/tools/gopls@latest
```

Learn more in the
[advanced installation instructions](doc/advanced.md#installing-unreleased-versions).

Learn more about gopls releases in the [release policy](doc/releases.md).

## Setting up your workspace

`gopls` supports both Go module, multi-module and GOPATH modes. See the
[workspace documentation](doc/workspace.md) for information on supported
workspace layouts.

## Configuration

You can configure `gopls` to change your editor experience or view additional
debugging information. Configuration options will be made available by your
editor, so see your [editor's instructions](#editors) for specific details. A
full list of `gopls` settings can be found in the [settings documentation](doc/settings.md).

### Environment variables

`gopls` inherits your editor's environment, so be aware of any environment
variables you configure. Some editors, such as VS Code, allow users to
selectively override the values of some environment variables.

## Support Policy

Gopls is maintained by engineers on the
[Go tools team](https://github.com/orgs/golang/teams/tools-team/members),
who actively monitor the
[Go](https://github.com/golang/go/issues?q=is%3Aissue+is%3Aopen+label%3Agopls)
and
[VS Code Go](https://github.com/golang/vscode-go/issues) issue trackers.

### Supported Go versions

`gopls` follows the
[Go Release Policy](https://golang.org/doc/devel/release.html#policy),
meaning that it officially supports the last 2 major Go releases. Per
[issue #39146](https://go.dev/issues/39146), we attempt to maintain best-effort
support for the last 4 major Go releases, but this support extends only to not
breaking the build and avoiding easily fixable regressions.

In the context of this discussion, gopls "supports" a Go version if it supports
being built with that Go version as well as integrating with the `go` command
of that Go version.

The following table shows the final gopls version that supports a given Go
version. Go releases more recent than any in the table can be used with any
version of gopls.

| Go Version  | Final gopls version with support (without warnings) |
| ----------- | --------------------------------------------------- |
| Go 1.12     | [gopls@v0.7.5](https://github.com/golang/tools/releases/tag/gopls%2Fv0.7.5) |
| Go 1.15     | [gopls@v0.9.5](https://github.com/golang/tools/releases/tag/gopls%2Fv0.9.5) |

Our extended support is enforced via [continuous integration with older Go
versions](doc/contributing.md#ci). This legacy Go CI may not block releases:
test failures may be skipped rather than fixed. Furthermore, if a regression in
an older Go version causes irreconcilable CI failures, we may drop support for
that Go version in CI if it is 3 or 4 Go versions old.

### Supported build systems

`gopls` currently only supports the `go` command, so if you are using
a different build system, `gopls` will not work well. Bazel is not officially
supported, but may be made to work with an appropriately configured
`go/packages` driver. See
[bazelbuild/rules_go#512](https://github.com/bazelbuild/rules_go/issues/512)
for more information.
You can follow [these instructions](https://github.com/bazelbuild/rules_go/wiki/Editor-setup)
to configure your `gopls` to work with Bazel.

### Troubleshooting

If you are having issues with `gopls`, please follow the steps described in the
[troubleshooting guide](doc/troubleshooting.md).

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
