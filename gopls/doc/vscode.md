# VSCode

Use the [VSCode-Go] plugin, with the following configuration:

```json5
"go.useLanguageServer": true,
"[go]": {
    "editor.snippetSuggestions": "none",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
},
"gopls": {
    "usePlaceholders": true, // add parameter placeholders when completing a function
    "wantCompletionDocumentation": true // for documentation in completion items
},
"files.eol": "\n", // formatting only supports LF line endings
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

See the [section on command line](user.md#command-line) arguments for more information about what these do, along with other things like `--logfile=auto` that you might want to use.

You can disable features through the `"go.languageServerExperimentalFeatures"` section of the config. An example of a feature you may want to disable is `"documentLink"`, which opens Godoc links when you click on import statements in your file.


[VSCode-Go]: https://github.com/microsoft/vscode-go
