# Documentation for contributors

This documentation augments the general documentation for contributing to the
x/tools repository, described at the [repository root](../../CONTRIBUTING.md).

Contributions are welcome, but since development is so active, we request that
you file an issue and claim it before starting to work on something. Otherwise,
it is likely that we might already be working on a fix for your issue.

## Finding issues

All `gopls` issues are labeled as such (see the [`gopls` label][issue-gopls]).
Issues that are suitable for contributors are additionally tagged with the
[`help-wanted` label][issue-wanted].

Before you begin working on an issue, please leave a comment that you are
claiming it.

## Getting started

Most of the `gopls` logic is in the `golang.org/x/tools/gopls/internal/lsp`
directory.

## Build

To build a version of `gopls` with your changes applied:

```bash
cd /path/to/tools/gopls
go install
```

To confirm that you are testing with the correct `gopls` version, check that
your `gopls` version looks like this:

```bash
$ gopls version
golang.org/x/tools/gopls master
    golang.org/x/tools/gopls@(devel)
```

## Getting help

The best way to contact the gopls team directly is via the
[#gopls-dev](https://app.slack.com/client/T029RQSE6/CRWSN9NCD) channel on the
gophers slack. Please feel free to ask any questions about your contribution or
about contributing in general.


## Error handling

It is important for the user experience that, whenever practical,
minor logic errors in a particular feature don't cause the server to
crash.

The representation of a Go program is complex. The import graph of
package metadata, the syntax trees of parsed files, and their
associated type information together form a huge API surface area.
Even when the input is valid, there are many edge cases to consider,
and this grows by an order of magnitude when you consider missing
imports, parse errors, and type errors.

What should you do when your logic must handle an error that you
believe "can't happen"?

- If it's possible to return an error, then use the `bug.Errorf`
  function to return an error to the user, but also record the bug in
  gopls' cache so that it is less likely to be ignored.

- If it's safe to proceed, you can call `bug.Reportf` to record the
  error and continue as normal.

- If there's no way to proceed, call `bug.Fatalf` to record the error
  and then stop the program with `log.Fatalf`. You can also use
  `bug.Panicf` if there's a chance that a recover handler might save
  the situation.

- Only if you can prove locally that an error is impossible should you
  call `log.Fatal`. If the error may happen for some input, however
  unlikely, then you should use one of the approaches above. Also, if
  the proof of safety depends on invariants broadly distributed across
  the code base, then you should instead use `bug.Panicf`.

Note also that panicking is preferable to `log.Fatal` because it
allows VS Code's crash reporting to recognize and capture the stack.

Bugs reported through `bug.Errorf` and friends are retrieved using the
`gopls bug` command, which opens a GitHub Issue template and populates
it with a summary of each bug and its frequency.
The text of the bug is rather fastidiously printed to stdout to avoid
sharing user names and error message strings (which could contain
project identifiers) with GitHub.
Users are invited to share it if they are willing.

## Testing

To run tests for just `gopls/`, run,

```bash
cd /path/to/tools/gopls
go test ./...
```

But, much of the gopls work involves `internal/lsp` too, so you will want to
run both:

```bash
cd /path/to/tools
cd gopls && go test ./...
cd ..
go test ./internal/lsp/...
```

There is additional information about the `internal/lsp` tests in the
[internal/lsp/tests `README`](https://github.com/golang/tools/blob/master/internal/lsp/tests/README.md).

### Regtests

gopls has a suite of regression tests defined in the `./gopls/internal/regtest`
directory. Each of these tests writes files to a temporary directory, starts a
separate gopls session, and scripts interactions using an editor-like API. As a
result of this overhead they can be quite slow, particularly on systems where
file operations are costly.

Due to the asynchronous nature of the LSP, regtests assertions are written
as 'expectations' that the editor state must achieve _eventually_. This can
make debugging the regtests difficult. To aid with debugging, the regtests
output their LSP logs on any failure. If your CL gets a test failure while
running the regtests, please do take a look at the description of the error and
the LSP logs, but don't hesitate to [reach out](#getting-help) to the gopls
team if you need help.

### CI

When you mail your CL and you or a fellow contributor assigns the
`Run-TryBot=1` label in Gerrit, the
[TryBots](https://golang.org/doc/contribute.html#trybots) will run tests in
both the `golang.org/x/tools` and `golang.org/x/tools/gopls` modules, as
described above.

Furthermore, an additional "gopls-CI" pass will be run by _Kokoro_, which is a
Jenkins-like Google infrastructure for running Dockerized tests. This allows us
to run gopls tests in various environments that would be difficult to add to
the TryBots. Notably, Kokoro runs tests on
[older Go versions](../README.md#supported-go-versions) that are no longer supported
by the TryBots. Per that that policy, support for these older Go versions is
best-effort, and test failures may be skipped rather than fixed.

Kokoro runs are triggered by the `Run-TryBot=1` label, just like TryBots, but
unlike TryBots they do not automatically re-run if the "gopls-CI" result is
removed in Gerrit. To force a re-run of the Kokoro CI on a CL containing the
`Run-TryBot=1` label, you can reply in Gerrit with the comment "kokoro rerun".

## Debugging

The easiest way to debug your change is to run a single `gopls` test with a
debugger.

See also [Troubleshooting](troubleshooting.md#troubleshooting).

<!--TODO(rstambler): Add more details about the debug server and viewing
telemetry.-->

[issue-gopls]: https://github.com/golang/go/issues?utf8=%E2%9C%93&q=is%3Aissue+is%3Aopen+label%3Agopls "gopls issues"
[issue-wanted]: https://github.com/golang/go/issues?utf8=✓&q=is%3Aissue+is%3Aopen+label%3Agopls+label%3A"help+wanted" "help wanted"
