# Sublime Text

Use the [LSP] package. After installing it using Package Control, do the following:

* Open the **Command Palette**
* Find and run the command **LSP: Enable Language Server Globally**
* Select the **gopls** item. Be careful not to select the similarly named *golsp* by mistake.

Finally, you should familiarise yourself with the LSP package's *Settings* and *Key Bindings*. Find them under the menu item **Preferences > Package Settings > LSP**.

## Examples
Minimal global LSP settings, that assume **gopls** and **go** appear on the PATH seen by Sublime Text:<br>
```
{
    "clients": {
        "gopls": {
             "enabled": true,
         }
    }
}
```

Global LSP settings that supply a specific PATH for finding **gopls** and **go**, as well as some settings for Sublime LSP itself:
```
{
    "clients": {
        "gopls": {
            "enabled": true,
            "env": {
                "PATH": "/path/to/your/go/bin",
            }
        }
    },
    // Recommended by https://agniva.me/gopls/2021/01/02/setting-up-gopls-sublime.html
    // except log_stderr mentioned there is no longer recognized.
    "show_references_in_quick_panel": true,
    "log_debug": true,
    // These two are recommended by LSP-json as replacement for deprecated only_show_lsp_completions
    "inhibit_snippet_completions": true,
    "inhibit_word_completions": true,
 }
 ```

LSP and gopls settings can also be adjusted on a per-project basis to override global settings.
```
{
    "folders": [
        {
            "path": "/path/to/a/folder/one"
        },
        {
            // If you happen to be working on Go itself, this can be helpful; go-dev/bin should be on PATH.
            "path": "/path/to/your/go-dev/src/cmd"
        }
     ],
    "settings": {
        "LSP": {
            "gopls": {
                "env": {
                    "PATH": "/path/to/your/go-dev/bin:/path/to/your/go/bin",
                    "GOPATH": "",
                },
                "settings": {
                    "experimentalWorkspaceModule": true
                }
            }
        },
        // This will apply for all languages in this project that have
        // LSP servers, not just Go, however cannot enable just for Go.
        "lsp_format_on_save": true,
    }
}
```

Usually changes to these settings are recognized after saving the project file, but it may sometimes be necessary to either restart the server(s) (**Tools > LSP > Restart Servers**) or quit and restart Sublime Text itself.

[LSP]: https://packagecontrol.io/packages/LSP
