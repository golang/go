// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cmdtest contains the test suite for the command line behavior of gopls.
package cmdtest

// This file defines integration tests of each gopls subcommand that
// fork+exec the command in a separate process.
//
// (Rather than execute 'go build gopls' during the test, we reproduce
// the main entrypoint in the test executable.)
//
// The purpose of this test is to exercise client-side logic such as
// argument parsing and formatting of LSP RPC responses, not server
// behavior; see lsp_test for that.
//
// All tests run in parallel.
//
// TODO(adonovan):
// - Use markers to represent positions in the input and in assertions.
// - Coverage of cross-cutting things like cwd, environ, span parsing, etc.
// - Subcommands that accept -write and -diff flags should implement
//   them consistently wrt the default behavior; factor their tests.
// - Add missing test for 'vulncheck' subcommand.
// - Add tests for client-only commands: serve, bug, help, api-json, licenses.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	exec "golang.org/x/sys/execabs"
	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/hooks"
	"golang.org/x/tools/gopls/internal/lsp/cmd"
	"golang.org/x/tools/gopls/internal/lsp/debug"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/internal/testenv"
	"golang.org/x/tools/internal/tool"
	"golang.org/x/tools/txtar"
)

// TestVersion tests the 'version' subcommand (../info.go).
func TestVersion(t *testing.T) {
	t.Parallel()

	tree := writeTree(t, "")

	// There's not much we can robustly assert about the actual version.
	const want = debug.Version // e.g. "master"

	// basic
	{
		res := gopls(t, tree, "version")
		res.checkExit(true)
		res.checkStdout(want)
	}

	// -json flag
	{
		res := gopls(t, tree, "version", "-json")
		res.checkExit(true)
		var v debug.ServerVersion
		if res.toJSON(&v) {
			if v.Version != want {
				t.Errorf("expected Version %q, got %q (%v)", want, v.Version, res)
			}
		}
	}
}

// TestCheck tests the 'check' subcommand (../check.go).
func TestCheck(t *testing.T) {
	t.Parallel()

	tree := writeTree(t, `
-- go.mod --
module example.com
go 1.18

-- a.go --
package a
import "fmt"
var _ = fmt.Sprintf("%s", 123)

-- b.go --
package a
import "fmt"
var _ = fmt.Sprintf("%d", "123")
`)

	// no files
	{
		res := gopls(t, tree, "check")
		res.checkExit(true)
		if res.stdout != "" {
			t.Errorf("unexpected output: %v", res)
		}
	}

	// one file
	{
		res := gopls(t, tree, "check", "./a.go")
		res.checkExit(true)
		res.checkStdout("fmt.Sprintf format %s has arg 123 of wrong type int")
	}

	// two files
	{
		res := gopls(t, tree, "check", "./a.go", "./b.go")
		res.checkExit(true)
		res.checkStdout(`a.go:.* fmt.Sprintf format %s has arg 123 of wrong type int`)
		res.checkStdout(`b.go:.* fmt.Sprintf format %d has arg "123" of wrong type string`)
	}
}

// TestCallHierarchy tests the 'call_hierarchy' subcommand (../call_hierarchy.go).
func TestCallHierarchy(t *testing.T) {
	t.Parallel()

	tree := writeTree(t, `
-- go.mod --
module example.com
go 1.18

-- a.go --
package a
func f() {}
func g() {
	f()
}
func h() {
	f()
	f()
}
`)
	// missing position
	{
		res := gopls(t, tree, "call_hierarchy")
		res.checkExit(false)
		res.checkStderr("expects 1 argument")
	}
	// wrong place
	{
		res := gopls(t, tree, "call_hierarchy", "a.go:1")
		res.checkExit(false)
		res.checkStderr("identifier not found")
	}
	// f is called once from g and twice from h.
	{
		res := gopls(t, tree, "call_hierarchy", "a.go:2:6")
		res.checkExit(true)
		// We use regexp '.' as an OS-agnostic path separator.
		res.checkStdout("ranges 7:2-3, 8:2-3 in ..a.go from/to function h in ..a.go:6:6-7")
		res.checkStdout("ranges 4:2-3 in ..a.go from/to function g in ..a.go:3:6-7")
		res.checkStdout("identifier: function f in ..a.go:2:6-7")
	}
}

// TestDefinition tests the 'definition' subcommand (../definition.go).
func TestDefinition(t *testing.T) {
	t.Parallel()

	tree := writeTree(t, `
-- go.mod --
module example.com
go 1.18

-- a.go --
package a
import "fmt"
func f() {
	fmt.Println()
}
func g() {
	f()
}
`)
	// missing position
	{
		res := gopls(t, tree, "definition")
		res.checkExit(false)
		res.checkStderr("expects 1 argument")
	}
	// intra-package
	{
		res := gopls(t, tree, "definition", "a.go:7:2") // "f()"
		res.checkExit(true)
		res.checkStdout("a.go:3:6-7: defined here as func f")
	}
	// cross-package
	{
		res := gopls(t, tree, "definition", "a.go:4:7") // "Println"
		res.checkExit(true)
		res.checkStdout("print.go.* defined here as func fmt.Println")
		res.checkStdout("Println formats using the default formats for its operands")
	}
	// -json and -markdown
	{
		res := gopls(t, tree, "definition", "-json", "-markdown", "a.go:4:7")
		res.checkExit(true)
		var defn cmd.Definition
		if res.toJSON(&defn) {
			if !strings.HasPrefix(defn.Description, "```go\nfunc fmt.Println") {
				t.Errorf("Description does not start with markdown code block. Got: %s", defn.Description)
			}
		}
	}
}

// TestFoldingRanges tests the 'folding_ranges' subcommand (../folding_range.go).
func TestFoldingRanges(t *testing.T) {
	t.Parallel()

	tree := writeTree(t, `
-- go.mod --
module example.com
go 1.18

-- a.go --
package a
func f(x int) {
	// hello
}
`)
	// missing filename
	{
		res := gopls(t, tree, "folding_ranges")
		res.checkExit(false)
		res.checkStderr("expects 1 argument")
	}
	// success
	{
		res := gopls(t, tree, "folding_ranges", "a.go")
		res.checkExit(true)
		res.checkStdout("2:8-2:13") // params (x int)
		res.checkStdout("2:16-4:1") //   body { ... }
	}
}

// TestFormat tests the 'format' subcommand (../format.go).
func TestFormat(t *testing.T) {
	t.Parallel()

	tree := writeTree(t, `
-- a.go --
package a ;  func f ( ) { }
`)
	const want = `package a

func f() {}
`

	// no files => nop
	{
		res := gopls(t, tree, "format")
		res.checkExit(true)
	}
	// default => print formatted result
	{
		res := gopls(t, tree, "format", "a.go")
		res.checkExit(true)
		if res.stdout != want {
			t.Errorf("format: got <<%s>>, want <<%s>>", res.stdout, want)
		}
	}
	// start/end position not supported (unless equal to start/end of file)
	{
		res := gopls(t, tree, "format", "a.go:1-2")
		res.checkExit(false)
		res.checkStderr("only full file formatting supported")
	}
	// -list: show only file names
	{
		res := gopls(t, tree, "format", "-list", "a.go")
		res.checkExit(true)
		res.checkStdout("a.go")
	}
	// -diff prints a unified diff
	{
		res := gopls(t, tree, "format", "-diff", "a.go")
		res.checkExit(true)
		// We omit the filenames as they vary by OS.
		want := `
-package a ;  func f ( ) { }
+package a
+
+func f() {}
`
		res.checkStdout(regexp.QuoteMeta(want))
	}
	// -write updates the file
	{
		res := gopls(t, tree, "format", "-write", "a.go")
		res.checkExit(true)
		res.checkStdout("^$") // empty
		checkContent(t, filepath.Join(tree, "a.go"), want)
	}
}

// TestHighlight tests the 'highlight' subcommand (../highlight.go).
func TestHighlight(t *testing.T) {
	t.Parallel()

	tree := writeTree(t, `
-- a.go --
package a
import "fmt"
func f() {
	fmt.Println()
	fmt.Println()
}
`)

	// no arguments
	{
		res := gopls(t, tree, "highlight")
		res.checkExit(false)
		res.checkStderr("expects 1 argument")
	}
	// all occurrences of Println
	{
		res := gopls(t, tree, "highlight", "a.go:4:7")
		res.checkExit(true)
		res.checkStdout("a.go:4:6-13")
		res.checkStdout("a.go:5:6-13")
	}
}

// TestImplementations tests the 'implementation' subcommand (../implementation.go).
func TestImplementations(t *testing.T) {
	t.Parallel()

	tree := writeTree(t, `
-- a.go --
package a
import "fmt"
type T int
func (T) String() string { return "" }
`)

	// no arguments
	{
		res := gopls(t, tree, "implementation")
		res.checkExit(false)
		res.checkStderr("expects 1 argument")
	}
	// T.String
	{
		res := gopls(t, tree, "implementation", "a.go:4:10")
		res.checkExit(true)
		// TODO(adonovan): extract and check the content of the reported ranges?
		// We use regexp '.' as an OS-agnostic path separator.
		res.checkStdout("fmt.print.go:")     // fmt.Stringer.String
		res.checkStdout("runtime.error.go:") // runtime.stringer.String
	}
}

// TestImports tests the 'imports' subcommand (../imports.go).
func TestImports(t *testing.T) {
	t.Parallel()

	tree := writeTree(t, `
-- a.go --
package a
func _() {
	fmt.Println()
}
`)

	want := `
package a

import "fmt"
func _() {
	fmt.Println()
}
`[1:]

	// no arguments
	{
		res := gopls(t, tree, "imports")
		res.checkExit(false)
		res.checkStderr("expects 1 argument")
	}
	// default: print with imports
	{
		res := gopls(t, tree, "imports", "a.go")
		res.checkExit(true)
		if res.stdout != want {
			t.Errorf("format: got <<%s>>, want <<%s>>", res.stdout, want)
		}
	}
	// -diff: show a unified diff
	{
		res := gopls(t, tree, "imports", "-diff", "a.go")
		res.checkExit(true)
		res.checkStdout(regexp.QuoteMeta(`+import "fmt"`))
	}
	// -write: update file
	{
		res := gopls(t, tree, "imports", "-write", "a.go")
		res.checkExit(true)
		checkContent(t, filepath.Join(tree, "a.go"), want)
	}
}

// TestLinks tests the 'links' subcommand (../links.go).
func TestLinks(t *testing.T) {
	t.Parallel()

	tree := writeTree(t, `
-- a.go --
// Link in package doc: https://pkg.go.dev/
package a

// Link in internal comment: https://go.dev/cl

// Doc comment link: https://blog.go.dev/
func f() {}
`)
	// no arguments
	{
		res := gopls(t, tree, "links")
		res.checkExit(false)
		res.checkStderr("expects 1 argument")
	}
	// success
	{
		res := gopls(t, tree, "links", "a.go")
		res.checkExit(true)
		res.checkStdout("https://go.dev/cl")
		res.checkStdout("https://pkg.go.dev")
		res.checkStdout("https://blog.go.dev/")
	}
	// -json
	{
		res := gopls(t, tree, "links", "-json", "a.go")
		res.checkExit(true)
		res.checkStdout("https://pkg.go.dev")
		res.checkStdout("https://go.dev/cl")
		res.checkStdout("https://blog.go.dev/") // at 5:21-5:41
		var links []protocol.DocumentLink
		if res.toJSON(&links) {
			// Check just one of the three locations.
			if got, want := fmt.Sprint(links[2].Range), "5:21-5:41"; got != want {
				t.Errorf("wrong link location: got %v, want %v", got, want)
			}
		}
	}
}

// TestReferences tests the 'references' subcommand (../references.go).
func TestReferences(t *testing.T) {
	t.Parallel()

	tree := writeTree(t, `
-- go.mod --
module example.com
go 1.18

-- a.go --
package a
import "fmt"
func f() {
	fmt.Println()
}

-- b.go --
package a
import "fmt"
func g() {
	fmt.Println()
}
`)
	// no arguments
	{
		res := gopls(t, tree, "references")
		res.checkExit(false)
		res.checkStderr("expects 1 argument")
	}
	// fmt.Println
	{
		res := gopls(t, tree, "references", "a.go:4:10")
		res.checkExit(true)
		res.checkStdout("a.go:4:6-13")
		res.checkStdout("b.go:4:6-13")
	}
}

// TestSignature tests the 'signature' subcommand (../signature.go).
func TestSignature(t *testing.T) {
	t.Parallel()

	tree := writeTree(t, `
-- go.mod --
module example.com
go 1.18

-- a.go --
package a
import "fmt"
func f() {
	fmt.Println(123)
}
`)
	// no arguments
	{
		res := gopls(t, tree, "signature")
		res.checkExit(false)
		res.checkStderr("expects 1 argument")
	}
	// at 123 inside fmt.Println() call
	{
		res := gopls(t, tree, "signature", "a.go:4:15")
		res.checkExit(true)
		res.checkStdout("Println\\(a ...")
		res.checkStdout("Println formats using the default formats...")
	}
}

// TestPrepareRename tests the 'prepare_rename' subcommand (../prepare_rename.go).
func TestPrepareRename(t *testing.T) {
	t.Parallel()

	tree := writeTree(t, `
-- go.mod --
module example.com
go 1.18

-- a.go --
package a
func oldname() {}
`)
	// no arguments
	{
		res := gopls(t, tree, "prepare_rename")
		res.checkExit(false)
		res.checkStderr("expects 1 argument")
	}
	// in 'package' keyword
	{
		res := gopls(t, tree, "prepare_rename", "a.go:1:3")
		res.checkExit(false)
		res.checkStderr("request is not valid at the given position")
	}
	// in 'package' identifier (not supported by client)
	{
		res := gopls(t, tree, "prepare_rename", "a.go:1:9")
		res.checkExit(false)
		res.checkStderr("can't rename package")
	}
	// in func oldname
	{
		res := gopls(t, tree, "prepare_rename", "a.go:2:9")
		res.checkExit(true)
		res.checkStdout("a.go:2:6-13") // all of "oldname"
	}
}

// TestRename tests the 'rename' subcommand (../rename.go).
func TestRename(t *testing.T) {
	t.Parallel()

	tree := writeTree(t, `
-- go.mod --
module example.com
go 1.18

-- a.go --
package a
func oldname() {}
`)
	// no arguments
	{
		res := gopls(t, tree, "rename")
		res.checkExit(false)
		res.checkStderr("expects 2 arguments")
	}
	// missing newname
	{
		res := gopls(t, tree, "rename", "a.go:1:3")
		res.checkExit(false)
		res.checkStderr("expects 2 arguments")
	}
	// in 'package' keyword
	{
		res := gopls(t, tree, "rename", "a.go:1:3", "newname")
		res.checkExit(false)
		res.checkStderr("no identifier found")
	}
	// in 'package' identifier
	{
		res := gopls(t, tree, "rename", "a.go:1:9", "newname")
		res.checkExit(false)
		res.checkStderr(`cannot rename package: module path .* same as the package path, so .* no effect`)
	}
	// success, func oldname (and -diff)
	{
		res := gopls(t, tree, "rename", "-diff", "a.go:2:9", "newname")
		res.checkExit(true)
		res.checkStdout(regexp.QuoteMeta("-func oldname() {}"))
		res.checkStdout(regexp.QuoteMeta("+func newname() {}"))
	}
}

// TestSymbols tests the 'symbols' subcommand (../symbols.go).
func TestSymbols(t *testing.T) {
	t.Parallel()

	tree := writeTree(t, `
-- go.mod --
module example.com
go 1.18

-- a.go --
package a
func f()
var v int
const c = 0
`)
	// no files
	{
		res := gopls(t, tree, "symbols")
		res.checkExit(false)
		res.checkStderr("expects 1 argument")
	}
	// success
	{
		res := gopls(t, tree, "symbols", "a.go:123:456") // (line/col ignored)
		res.checkExit(true)
		res.checkStdout("f Function 2:6-2:7")
		res.checkStdout("v Variable 3:5-3:6")
		res.checkStdout("c Constant 4:7-4:8")
	}
}

// TestSemtok tests the 'semtok' subcommand (../semantictokens.go).
func TestSemtok(t *testing.T) {
	t.Parallel()

	tree := writeTree(t, `
-- go.mod --
module example.com
go 1.18

-- a.go --
package a
func f()
var v int
const c = 0
`)
	// no files
	{
		res := gopls(t, tree, "semtok")
		res.checkExit(false)
		res.checkStderr("expected one file name")
	}
	// success
	{
		res := gopls(t, tree, "semtok", "a.go")
		res.checkExit(true)
		got := res.stdout
		want := `
/*⇒7,keyword,[]*/package /*⇒1,namespace,[]*/a
/*⇒4,keyword,[]*/func /*⇒1,function,[definition]*/f()
/*⇒3,keyword,[]*/var /*⇒1,variable,[definition]*/v /*⇒3,type,[defaultLibrary]*/int
/*⇒5,keyword,[]*/const /*⇒1,variable,[definition readonly]*/c = /*⇒1,number,[]*/0
`[1:]
		if got != want {
			t.Errorf("semtok: got <<%s>>, want <<%s>>", got, want)
		}
	}
}

func TestStats(t *testing.T) {
	t.Parallel()

	tree := writeTree(t, `
-- go.mod --
module example.com
go 1.18

-- a.go --
package a
-- b/b.go --
package b
-- testdata/foo.go --
package foo
`)

	// Trigger a bug report with a distinctive string
	// and check that it was durably recorded.
	oops := fmt.Sprintf("oops-%d", rand.Int())
	{
		env := []string{"TEST_GOPLS_BUG=" + oops}
		res := goplsWithEnv(t, tree, env, "bug")
		res.checkExit(true)
	}

	res := gopls(t, tree, "stats")
	res.checkExit(true)

	var stats cmd.GoplsStats
	if err := json.Unmarshal([]byte(res.stdout), &stats); err != nil {
		t.Fatalf("failed to unmarshal JSON output of stats command: %v", err)
	}

	// a few sanity checks
	checks := []struct {
		field string
		got   int
		want  int
	}{
		{
			"WorkspaceStats.Views[0].WorkspaceModules",
			stats.WorkspaceStats.Views[0].WorkspacePackages.Modules,
			1,
		},
		{
			"WorkspaceStats.Views[0].WorkspacePackages",
			stats.WorkspaceStats.Views[0].WorkspacePackages.Packages,
			2,
		},
		{"DirStats.Files", stats.DirStats.Files, 4},
		{"DirStats.GoFiles", stats.DirStats.GoFiles, 2},
		{"DirStats.ModFiles", stats.DirStats.ModFiles, 1},
		{"DirStats.TestdataFiles", stats.DirStats.TestdataFiles, 1},
	}
	for _, check := range checks {
		if check.got != check.want {
			t.Errorf("stats.%s = %d, want %d", check.field, check.got, check.want)
		}
	}

	// Check that we got a BugReport with the expected message.
	{
		got := fmt.Sprint(stats.BugReports)
		wants := []string{
			"cmd/info.go", // File containing call to bug.Report
			oops,          // Description
		}
		for _, want := range wants {
			if !strings.Contains(got, want) {
				t.Errorf("BugReports does not contain %q. Got:<<%s>>", want, got)
				break
			}
		}
	}

	// Check that -anon suppresses fields containing user information.
	{
		res2 := gopls(t, tree, "stats", "-anon")
		res2.checkExit(true)
		var stats2 cmd.GoplsStats
		if err := json.Unmarshal([]byte(res2.stdout), &stats2); err != nil {
			t.Fatalf("failed to unmarshal JSON output of stats command: %v", err)
		}
		if got := len(stats2.BugReports); got > 0 {
			t.Errorf("Got %d bug reports with -anon, want 0. Reports:%+v", got, stats2.BugReports)
		}
	}
}

// TestFix tests the 'fix' subcommand (../suggested_fix.go).
func TestFix(t *testing.T) {
	t.Parallel()

	tree := writeTree(t, `
-- go.mod --
module example.com
go 1.18

-- a.go --
package a
type T int
func f() (int, string) { return }
`)
	want := `
package a
type T int
func f() (int, string) { return 0, "" }
`[1:]

	// no arguments
	{
		res := gopls(t, tree, "fix")
		res.checkExit(false)
		res.checkStderr("expects at least 1 argument")
	}
	// success (-a enables fillreturns)
	{
		res := gopls(t, tree, "fix", "-a", "a.go")
		res.checkExit(true)
		got := res.stdout
		if got != want {
			t.Errorf("fix: got <<%s>>, want <<%s>>\nstderr:\n%s", got, want, res.stderr)
		}
	}
	// TODO(adonovan): more tests:
	// - -write, -diff: factor with imports, format, rename.
	// - without -all flag
	// - args[2:] is an optional list of protocol.CodeActionKind enum values.
	// - a span argument with a range causes filtering.
}

// TestWorkspaceSymbol tests the 'workspace_symbol' subcommand (../workspace_symbol.go).
func TestWorkspaceSymbol(t *testing.T) {
	t.Parallel()

	tree := writeTree(t, `
-- go.mod --
module example.com
go 1.18

-- a.go --
package a
func someFunctionName()
`)
	// no files
	{
		res := gopls(t, tree, "workspace_symbol")
		res.checkExit(false)
		res.checkStderr("expects 1 argument")
	}
	// success
	{
		res := gopls(t, tree, "workspace_symbol", "meFun")
		res.checkExit(true)
		res.checkStdout("a.go:2:6-22 someFunctionName Function")
	}
}

// -- test framework --

func TestMain(m *testing.M) {
	switch os.Getenv("ENTRYPOINT") {
	case "goplsMain":
		goplsMain()
	default:
		os.Exit(m.Run())
	}
}

// This function is a stand-in for gopls.main in ../../../../main.go.
func goplsMain() {
	// Panic on bugs (unlike the production gopls command),
	// except in tests that inject calls to bug.Report.
	if os.Getenv("TEST_GOPLS_BUG") == "" {
		bug.PanicOnBugs = true
	}

	tool.Main(context.Background(), cmd.New("gopls", "", nil, hooks.Options), os.Args[1:])
}

// writeTree extracts a txtar archive into a new directory and returns its path.
func writeTree(t *testing.T, archive string) string {
	root := t.TempDir()

	// This unfortunate step is required because gopls output
	// expands symbolic links it its input file names (arguably it
	// should not), and on macOS the temp dir is in /var -> private/var.
	root, err := filepath.EvalSymlinks(root)
	if err != nil {
		t.Fatal(err)
	}

	for _, f := range txtar.Parse([]byte(archive)).Files {
		filename := filepath.Join(root, f.Name)
		if err := os.MkdirAll(filepath.Dir(filename), 0777); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filename, f.Data, 0666); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

// gopls executes gopls in a child process.
func gopls(t *testing.T, dir string, args ...string) *result {
	return goplsWithEnv(t, dir, nil, args...)
}

func goplsWithEnv(t *testing.T, dir string, env []string, args ...string) *result {
	testenv.NeedsTool(t, "go")

	// Catch inadvertent use of dir=".", which would make
	// the ReplaceAll below unpredictable.
	if !filepath.IsAbs(dir) {
		t.Fatalf("dir is not absolute: %s", dir)
	}

	goplsCmd := exec.Command(os.Args[0], args...)
	goplsCmd.Env = append(os.Environ(), "ENTRYPOINT=goplsMain")
	goplsCmd.Env = append(goplsCmd.Env, env...)
	goplsCmd.Dir = dir
	goplsCmd.Stdout = new(bytes.Buffer)
	goplsCmd.Stderr = new(bytes.Buffer)

	cmdErr := goplsCmd.Run()

	stdout := strings.ReplaceAll(fmt.Sprint(goplsCmd.Stdout), dir, ".")
	stderr := strings.ReplaceAll(fmt.Sprint(goplsCmd.Stderr), dir, ".")
	exitcode := 0
	if cmdErr != nil {
		if exitErr, ok := cmdErr.(*exec.ExitError); ok {
			exitcode = exitErr.ExitCode()
		} else {
			stderr = cmdErr.Error() // (execve failure)
			exitcode = -1
		}
	}
	res := &result{
		t:        t,
		command:  "gopls " + strings.Join(args, " "),
		exitcode: exitcode,
		stdout:   stdout,
		stderr:   stderr,
	}
	if false {
		t.Log(res)
	}
	return res
}

// A result holds the result of a gopls invocation, and provides assertion helpers.
type result struct {
	t              *testing.T
	command        string
	exitcode       int
	stdout, stderr string
}

func (res *result) String() string {
	return fmt.Sprintf("%s: exit=%d stdout=<<%s>> stderr=<<%s>>",
		res.command, res.exitcode, res.stdout, res.stderr)
}

// checkExit asserts that gopls returned the expected exit code.
func (res *result) checkExit(success bool) {
	res.t.Helper()
	if (res.exitcode == 0) != success {
		res.t.Errorf("%s: exited with code %d, want success: %t (%s)",
			res.command, res.exitcode, success, res)
	}
}

// checkStdout asserts that the gopls standard output matches the pattern.
func (res *result) checkStdout(pattern string) {
	res.t.Helper()
	res.checkOutput(pattern, "stdout", res.stdout)
}

// checkStderr asserts that the gopls standard error matches the pattern.
func (res *result) checkStderr(pattern string) {
	res.t.Helper()
	res.checkOutput(pattern, "stderr", res.stderr)
}

func (res *result) checkOutput(pattern, name, content string) {
	res.t.Helper()
	if match, err := regexp.MatchString(pattern, content); err != nil {
		res.t.Errorf("invalid regexp: %v", err)
	} else if !match {
		res.t.Errorf("%s: %s does not match [%s]; got <<%s>>",
			res.command, name, pattern, content)
	}
}

// toJSON decodes res.stdout as JSON into to *ptr and reports its success.
func (res *result) toJSON(ptr interface{}) bool {
	if err := json.Unmarshal([]byte(res.stdout), ptr); err != nil {
		res.t.Errorf("invalid JSON %v", err)
		return false
	}
	return true
}

// checkContent checks that the contents of the file are as expected.
func checkContent(t *testing.T, filename, want string) {
	data, err := os.ReadFile(filename)
	if err != nil {
		t.Error(err)
		return
	}
	if got := string(data); got != want {
		t.Errorf("content of %s is <<%s>>, want <<%s>>", filename, got, want)
	}
}
