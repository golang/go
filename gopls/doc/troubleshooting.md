# Troubleshooting

If you suspect that `gopls` is crashing or not working correctly, please follow the [troubleshooting steps](#steps) below.

If `gopls` is using too much memory, please follow the steps under [Memory usage](#memory-usage).

## Steps

<!--- TODO: troubleshooting
describe more basic and optional trouble shooting steps
  like checking you opened the module root
  and using the debug pages
--->

1. Make sure your `gopls` is [up to date](user.md#installing).
1. Check the [known issues](status.md#known-issues).
1. [Report the issue](#file-an-issue).

## File an issue

You can use:

* Your editor's bug submission integration (if available). For instance, `:GoReportGitHubIssue` in [`vim-go`](vim.md#vim-go).
* `gopls bug` on the command line.
* The [Go issue tracker](https://github.com/golang/go/issues/new?title=x%2Ftools%2Fgopls%3A%20%3Cfill%20this%20in%3E).

Along with an explanation of the issue, please share the information listed here:

1. Your editor and any settings you have configured (for example, your VSCode `settings.json` file).
1. A sample program that reproduces the issue, if possible.
1. The output of `gopls version` on the command line.
1. The output of `gopls -rpc.trace -v check /path/to/file.go`.
1. gopls logs from when the issue occurred, as well as a timestamp for when the issue began to occur. See the [instructions](#capturing-gopls-logs) for information on how to capture gopls logs.

Much of this information is filled in for you if you use `gopls bug` to file the issue.

### Capturing logs

#### VS Code

For VSCode users, the gopls log can be found by navigating to `View` -> `Output` (or `Ctrl+K Ctrl+H`). There will be a drop-down menu titled `Tasks` in the top-right corner. Select the `gopls (server)` item, which will contain the `gopls` logs.

To increase the level of detail in your logs, add the following to your VS Code settings:

```json5
"go.languageServerFlags": [
  "-rpc.trace"
]
```

To start a debug server that will allow you to see profiles and memory usage, add the following to your VS Code settings:

```json5
"go.languageServerFlags": [
  "serve",
  "-rpc.trace",
  "--debug=localhost:6060",
],
```

You will then be able to view debug information by navigating to `localhost:6060`.

#### Other editors

For other editors, you may have to directly pass a `-logfile` flag to gopls.

To increase the level of detail in your logs, start `gopls` with the `-rpc.trace` flag. To start a debug server that will allow you to see profiles and memory usage, start `gopls` with `serve --debug=localhost:6060`. You will then be able to view debug information by navigating to `localhost:6060`.

If you are unsure of how to pass a flag to `gopls` through your editor, please see the [documentation for your editor](user.md#editors).

### Restart your editor

Once you have filed an issue, you can then try to restart your `gopls` instance by restarting your editor. In many cases, this will correct the problem. In VSCode, the easiest way to restart the language server is by opening the command palette (Ctrl + Shift + P) and selecting `"Go: Restart Language Server"`. You can also reload the VSCode instance by selecting `"Developer: Reload Window"`.

## Memory usage

`gopls` automatically writes out memory debug information when your usage
exceeds 1GB. This information can be found in your temporary directory with
names like `gopls.1234-5GiB-withnames.zip`. On Windows, your temporary
directory will be located at `%TMP%`, and on Unixes, it will be `$TMPDIR`,
which is usually `/tmp`. Please create a
[new issue](https://github.com/golang/go/issues/new?title=x%2Ftools%2Fgopls%3A%20%3Cfill%20this%20in%3E)
with your editor settings and memory debug information attached. If you are
uncomfortable sharing the package names of your code, you can share the
`-nonames` zip instead.
