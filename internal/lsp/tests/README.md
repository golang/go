## Testing

LSP has "marker tests" defined in `internal/lsp/testdata`, as well as
traditional tests.

#### Marker tests

Marker tests have a standard input file, like
`internal/lsp/testdata/foo/bar.go`, and some may have a corresponding golden
file, like `internal/lsp/testdata/foo/bar.go.golden`. The former is the "input"
and the latter is the expected output.

Each input file contains annotations like
`//@suggestedfix("}", "refactor.rewrite")`. These annotations are interpreted by
test runners to perform certain actions. The expected output after those actions
is encoded in the golden file.

When tests are run, each annotation results in a new subtest, which is encoded
in the golden file with a heading like,

```
-- suggestedfix_bar_11_21 --
// expected contents go here
-- suggestedfix_bar_13_20 --
// expected contents go here
```

The format of these headings vary: they are defined by the "Golden" function for
each annotation: https://pkg.go.dev/golang.org/x/tools/internal/lsp/tests#Data.Golden.
In the case above, the format is: annotation name, file name, annotation line
location, annotation character location.

So, if `internal/lsp/testdata/foo/bar.go` has three `suggestedfix` annotations,
the golden file should have three headers with `suggestedfix_bar_xx_yy`
headings.

To see a list of all available annotations, see the exported "expectations" in
[tests.go](https://github.com/golang/tools/blob/299f270db45902e93469b1152fafed034bb3f033/internal/lsp/tests/tests.go#L418-L447).

To run marker tests,

```
cd /path/to/tools

# The marker tests are located in "internal/lsp", "internal/lsp/cmd, and
# "internal/lsp/source".
go test ./internal/lsp/...
```

There are quite a lot of marker tests, so to run one individually, pass the test
path and heading into a -run argument:

```
cd /path/to/tools
go test ./internal/lsp -v -run TestLSP/Modules/SuggestedFix/bar_11_21
```

#### Resetting marker tests

Sometimes, a change is made to lsp that requires a change to multiple golden
files. When this happens, you can run,

```
cd /path/to/tools
./internal/lsp/reset_golden.sh
```