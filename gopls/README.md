# gopls documentation

[![PkgGoDev](https://pkg.go.dev/badge/golang.org/x/tools/gopls)](https://pkg.go.dev/golang.org/x/tools/gopls)

gopls (pronounced: "go please") is the official [language server] for the Go language.

## Status

It is currently in **alpha**, so it is **not stable**.

You can see more information about the status of gopls and its supported features [here](doc/status.md).

## Roadmap

The current goal is a fully stable build with the existing feature set, aiming
for the first half of 2020, with release candidates earlier in the year.

This will be the first build that we recommend people use, and will be tagged as the 1.0 version.
You can see the set of things being worked on in the [1.0 milestone], in general
we are focused on stability, specifically, making sure we have a reliable service that produces an experience in module mode that is not a retrograde step from the old tools in GOPATH mode.

There is also considerable effort being put into testing in order to make sure that we both have a stable service and also that we do not regress after launch.

While we may continue to accept contributions for new features, they may be turned off behind a configuration flag if they are not yet stable. See the [gopls unplanned] milestone for deprioritized features.

This is just a milestone for gopls itself. We work with editor integrators to make sure they can use the latest builds of gopls, and will help them use the 1.0 version as soon as it is ready, but that does not imply anything about the stability, supported features or version of the plugins.

## Using

In general you should not need to know anything about gopls, it should be integrated into your editor for you.

To install for your specific editor you can follow the following instructions

* [VSCode](doc/vscode.md)
* [Vim / Neovim](doc/vim.md)
* [Emacs](doc/emacs.md)
* [Acme](doc/acme.md)
* [Sublime Text](doc/subl.md)
* [Atom](doc/atom.md)

See the [user guide](doc/user.md) for more information, including the how to install gopls by hand if you need.

## Issues

If you are having issues with gopls, please first check the [known issues](doc/status.md#known-issues) before following the [troubleshooting](doc/troubleshooting.md#steps) guide.
If that does not give you the information you need, reach out to us.

You can chat with us on:
* the golang-tools [mailing list]
* the #gopls [slack channel] on the gophers slack

If you think you have an issue that needs fixing, or a feature suggestion, then please make sure you follow the steps to [file an issue](doc/troubleshooting.md#file-an-issue) with the right information to allow us to address it.

If you need to talk to us directly (for instance to file an issue with confidential information in it) you can reach out directly to [@stamblerre] or [@ianthehat].

## More information

If you want to know more about it, have an unusual use case, or want to contribute, please read the following documents

* [Using gopls](doc/user.md)
* [Troubleshooting and reporting issues](doc/troubleshooting.md)
* [Integrating gopls with an editor](doc/integrating.md)
* [Contributing to gopls](doc/contributing.md)
* [Design requirements and decisions](doc/design.md)
* [Implementation details](doc/implementation.md)

[language server]: https://langserver.org
[mailing list]: https://groups.google.com/forum/#!forum/golang-tools
[slack channel]: https://gophers.slack.com/messages/CJZH85XCZ
[@stamblerre]: https://github.com/stamblerre "Rebecca Stambler"
[@ianthehat]: https://github.com/ianthehat "Ian Cottrell"
[1.0 milestone]: https://github.com/golang/go/milestone/112
[gopls unplanned]: https://github.com/golang/go/milestone/124
