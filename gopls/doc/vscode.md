# VS Code

Use the [VS Code Go] plugin, with the following configuration:

```json5
"go.useLanguageServer": true,
```

As of February 2020, `gopls` will be enabled by default in [VS Code Go].
To learn more, follow along with
[golang.vscode-go#1037](https://github.com/golang/vscode-go/issues/1037).

```json5
"gopls": {
    // Add parameter placeholders when completing a function.
    "ui.completion.usePlaceholders": true,

    // If true, enable additional analyses with staticcheck.
    // Warning: This will significantly increase memory usage.
    "ui.diagnostic.staticcheck": false,
    
    // For more customization, see
    // see https://github.com/golang/vscode-go/blob/master/docs/settings.md.
}
```

To enable more detailed debug information, add the following to your VSCode settings:

```json5
"go.languageServerFlags": [
    "-rpc.trace", // for more detailed debug logging
    "serve",
    "--debug=localhost:6060", // Optional: investigate memory usage, see profiles
],
```

See the section on [command line](command-line.md) arguments for more
information about what these do, along with other things like
`--logfile=auto` that you might want to use.

## Build tags and flags

Build tags and flags will be automatically picked up from `"go.buildTags"` and
`"go.buildFlags"` settings. In the rare case that you don't want that default
behavior, you can still override the settings from the `gopls` section, using
`"gopls": { "build.buildFlags": [] }`.

## Remote Development with `gopls`

You can also make use of `gopls` with the
[VS Code Remote Development](https://code.visualstudio.com/docs/remote/remote-overview)
extensions to enable full-featured Go development on a lightweight client
machine, while connected to a more powerful server machine.

First, install the Remote Development extension of your choice, such as the
[Remote - SSH](https://code.visualstudio.com/docs/remote/ssh) extension. Once
you open a remote session in a new window, open the Extensions pane
(Ctrl+Shift+X) and you will see several different sections listed. In the
"Local - Installed" section, navigate to the Go extension and click
"Install in SSH: hostname".

Once you have reloaded VS Code, you will be prompted to install `gopls` and other
Go-related tools. After one more reload, you should be ready to develop remotely
with VS Code and the Go extension.

[VS Code Go]: https://github.com/golang/vscode-go
