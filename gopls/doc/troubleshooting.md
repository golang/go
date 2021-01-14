# Troubleshooting

If you suspect that `gopls` is crashing or not working correctly, please follow the troubleshooting steps below.

If `gopls` is using too much memory, please follow the steps under [Memory usage](#debug-memory-usage).

## Steps

VS Code users should follow [their troubleshooting guide](https://github.com/golang/vscode-go/blob/master/docs/troubleshooting.md), which has more a more specific version of these instructions.

1. Verify that your project is in good shape by working with it outside of your editor. Running a command like `go build ./...` in the workspace directory will compile everything. For modules, `go mod tidy` is another good check, though it may modify your `go.mod`.
1. Check that your editor isn't showing any diagnostics that indicate a problem with your workspace. They may appear as diagnostics on a Go file's package declaration, diagnostics in a go.mod file, or as a status or progress message. Problems in the workspace configuration can cause many different symptoms. See the [workspace setup instructions](workspace.md) for help.
1. Make sure `gopls` is up to date by following the [installation instructions](user.md#installing), then [restarting gopls](#restart-gopls).
1. Optionally, [ask for help](#ask-for-help) on Gophers Slack.
1. Finally, [report the issue](#file-an-issue) to the `gopls` developers.

## Restart `gopls`

`gopls` has no persistent state, so restarting it will fix transient problems. This is good and bad: good, because you can keep working, and bad, because you won't be able to debug the issue until it recurs.

In most cases, closing all your open editors will guarantee that `gopls` is killed and restarted. If you don't want to do that, there may be an editor command you can use to restart only `gopls`. Note that some `vim` configurations keep the server alive for a while after the editor exits; you may need to explicitly kill `gopls` if you use `vim`.

## Ask for help

Gophers Slack has active editor-specific channels like [#emacs](https://gophers.slack.com/archives/C0HKHULEM), [#vim](https://gophers.slack.com/archives/C07GBR52P), and [#vscode](https://gophers.slack.com/archives/C2B4L99RS) that can help debug further. If you're confident the problem is with `gopls`, you can go straight to [#gopls](https://gophers.slack.com/archives/CJZH85XCZ). Invites are [available to everyone](https://invite.slack.golangbridge.org). Come prepared with a short description of the issue, and try to be available to answer questions for a while afterward.

## File an issue

We can't diagnose a problem from just a description. When filing an issue, please include as much as possible of the following information:

1. Your editor and any settings you have configured (for example, your VSCode `settings.json` file).
1. A sample program that reproduces the issue, if possible.
1. The output of `gopls version` on the command line.
1. A complete gopls log file from a session where the issue occurred. It should have a `go env for <workspace folder>` log line near the beginning. It's also helpful to tell us the timestamp the problem occurred, so we can find it the log. See the [instructions](#capture-logs) for information on how to capture gopls logs.

Your editor may have a command that fills out some of the necessary information, such as `:GoReportGitHubIssue` in `vim-go`. Otherwise, you can use `gopls bug` on the command line. If neither of those work you can start from scratch directly on the [Go issue tracker](https://github.com/golang/go/issues/new?title=x%2Ftools%2Fgopls%3A%20%3Cfill%20this%20in%3E).

## Capture logs

You may have to change your editor's configuration to pass a `-logfile` flag to gopls.

To increase the level of detail in your logs, start `gopls` with the `-rpc.trace` flag. To start a debug server that will allow you to see profiles and memory usage, start `gopls` with `serve --debug=localhost:6060`. You will then be able to view debug information by navigating to `localhost:6060`.

If you are unsure of how to pass a flag to `gopls` through your editor, please see the [documentation for your editor](user.md#editors).

## Debug memory usage

`gopls` automatically writes out memory debug information when your usage exceeds 1GB. This information can be found in your temporary directory with names like `gopls.1234-5GiB-withnames.zip`. On Windows, your temporary directory will be located at `%TMP%`, and on Unixes, it will be `$TMPDIR`, which is usually `/tmp`. Please [file an issue](#file-an-issue) with this memory debug information attached. If you are uncomfortable sharing the package names of your code, you can share the `-nonames` zip instead, but it's much less useful.
