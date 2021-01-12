# VSCode

Use the [VSCode-Go] plugin, with the following configuration:

```json5
"go.useLanguageServer": true,
"[go]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true,
    },
    // Optional: Disable snippets, as they conflict with completion ranking.
    "editor.snippetSuggestions": "none",
},
"[go.mod]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true,
    },
},
"gopls": {
    // Add parameter placeholders when completing a function.
    "ui.completion.usePlaceholders": true,

    // If true, enable additional analyses with staticcheck.
    // Warning: This will significantly increase memory usage.
    "ui.diagnostic.staticcheck": false,
    
    // For more customization, see
    // see https://github.com/golang/vscode-go/blob/master/docs/settings.md
}
```

To enable more detailed debug information, add the following to your VSCode settings:

```json5
"go.languageServerFlags": [
    "-rpc.trace", // for more detailed debug logging
    "serve",
    "--debug=localhost:6060", // Optional: to investigate memory usage, see profiles
],
```

See the section on [command line](command-line.md) arguments for more information about what these do, along with other things like `--logfile=auto` that you might want to use.

You can disable features through the `"go.languageServerExperimentalFeatures"` section of the config. An example of a feature you may want to disable is `"documentLink"`, which opens [`pkg.go.dev`](https://pkg.go.dev) links when you click on import statements in your file.

### Build tags and flags

build tags and flags will be automatically picked up from `"go.buildTags"` and `"go.buildFlags"` settings. In rare cases if you don't want the default behavior, you can still override the settings from the `gopls` section, using `"gopls": { "build.buildFlags": [] }`.


[VSCode-Go]: https://github.com/golang/vscode-go

# VSCode Remote Development with gopls

You can also make use of `gopls` with the [VSCode Remote Development](https://code.visualstudio.com/docs/remote/remote-overview) extensions to enable full-featured Go development on a lightweight client machine, while connected to a more powerful server machine.

First, install the Remote Development extension of your choice, such as the [Remote - SSH](https://code.visualstudio.com/docs/remote/ssh) extension. Once you open a remote session in a new window, open the Extensions pane (Ctrl+Shift+X) and you will see several different sections listed. In the "Local - Installed" section, navigate to the Go extension and click "Install in SSH: hostname".

Once you have reloaded VSCode, you will be prompted to install `gopls` and other Go-related tools. After one more reload, you should be ready to develop remotely with VSCode and the Go extension.
