# VSCode

Use the [VSCode-Go] plugin, with the following configuration:

```json5
"go.useLanguageServer": true,
"[go]": {
    "editor.snippetSuggestions": "none",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true,
    }
},
"gopls": {
    "usePlaceholders": true, // add parameter placeholders when completing a function

    // Experimental settings
    "completeUnimported": true, // autocomplete unimported packages
    "deepCompletion": true,     // enable deep completion
}
```

VSCode will complain about the `"gopls"` settings, but they will still work. Once we have a consistent set of settings, we will make the changes in the VSCode plugin necessary to remove the errors.

If you encounter problems with import organization, please try setting a higher code action timeout (any value greater than 750ms), for example:

```json5
"[go]": {
  "editor.codeActionsOnSaveTimeout": 3000
}
```

To enable more detailed debug information, add the following to your VSCode settings:

```json5
"go.languageServerFlags": [
    "-rpc.trace", // for more detailed debug logging
    "serve",
    "--debug=localhost:6060", // to investigate memory usage, see profiles
],
```

See the section on [command line](command-line.md) arguments for more information about what these do, along with other things like `--logfile=auto` that you might want to use.

You can disable features through the `"go.languageServerExperimentalFeatures"` section of the config. An example of a feature you may want to disable is `"documentLink"`, which opens [`godoc.org`](https://godoc.org) links when you click on import statements in your file.


[VSCode-Go]: https://github.com/microsoft/vscode-go

# VSCode Remote Development with gopls

You can also make use of `gopls` with the [VSCode Remote Development](https://code.visualstudio.com/docs/remote/remote-overview) extensions to enable full-featured Go development on a lightweight client machine, while connected to a more powerful server machine.

First, install the Remote Development extension of your choice, such as the [Remote - SSH](https://code.visualstudio.com/docs/remote/ssh) extension. Once you open a remote session in a new window, open the Extensions pane (Ctrl+Shift+X) and you will see several different sections listed. In the "Local - Installed" section, navigate to the Go extension and click "Install in SSH: hostname".

Once you have reloaded VSCode, you will be prompted to install `gopls` and other Go-related tools. After one more reload, you should be ready to develop remotely with VSCode and the Go extension.
