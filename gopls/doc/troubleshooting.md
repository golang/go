# Troubleshooting

If you see a gopls error or crash, or gopls just stops working, please follow the troubleshooting steps below.

## Steps

<!--- TODO: troubleshooting
describe more basic and optional trouble shooting steps
  like checking you opened the module root
  and using the debug pages
--->

1. Make sure your `gopls` is [up to date](user.md#installing).
1. Check the [known issues](status.md#known-issues).
1. [Report the issue](file-an-issue).

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

For VSCode users, the gopls log can be found by going to `"View: Debug Console" -> "Output" -> "Tasks" -> "gopls"`. For other editors, you may have to directly pass a `-logfile` flag to gopls.

To increase the level of detail in your logs, start `gopls` with the `-rpc.trace` flag. To start a debug server that will allow you to see profiles and memory usage, start `gopls` with `serve --debug=localhost:6060`.

If you are unsure of how to pass a flag to `gopls` through your editor, please see the [documentation for your editor](user.md#editors).

### Restart your editor

Once you have filed an issue, you can then try to restart your `gopls` instance by restarting your editor. In many cases, this will correct the problem. In VSCode, the easiest way to restart the language server is by opening the command palette (Ctrl + Shift + P) and selecting `"Go: Restart Language Server"`. You can also reload the VSCode instance by selecting `"Developer: Reload Window"`.
