// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.11

package packages_test

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"sync"
	"testing"

	"golang.org/x/tools/go/packages"
)

// TODO(adonovan): more test cases to write:
//
// - When the tests fail, make them print a 'cd & load' command
//   that will allow the maintainer to interact with the failing scenario.
// - vendoring
// - errors in go-list metadata
// - all returned file names are absolute
// - a foo.test package that cannot be built for some reason (e.g.
//   import error) will result in a JSON blob with no name and a
//   nonexistent testmain file in GoFiles. Test that we handle this
//   gracefully.
// - IsTest boolean
//
// TypeCheck & WholeProgram modes:
//   - Fset may be user-supplied or not.
//   - Packages.Info is correctly set.
//   - typechecker configuration is honored
//   - import cycles are gracefully handled in type checker.
//   - test typechecking of generated test main and cgo.

func TestMetadataImportGraph(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skipf("TODO: skipping on non-Linux; fix this test to run everywhere. golang.org/issue/26387")
	}
	tmp, cleanup := makeTree(t, map[string]string{
		"src/a/a.go":             `package a; const A = 1`,
		"src/b/b.go":             `package b; import ("a"; _ "errors"); var B = a.A`,
		"src/c/c.go":             `package c; import (_ "b"; _ "unsafe")`,
		"src/c/c2.go":            "// +build ignore\n\n" + `package c; import _ "fmt"`,
		"src/subdir/d/d.go":      `package d`,
		"src/subdir/d/d_test.go": `package d; import _ "math/bits"`,
		"src/subdir/d/x_test.go": `package d_test; import _ "subdir/d"`, // TODO(adonovan): test bad import here
		"src/subdir/e/d.go":      `package e`,
		"src/e/e.go":             `package main; import _ "b"`,
		"src/e/e2.go":            `package main; import _ "c"`,
		"src/f/f.go":             `package f`,
	})
	defer cleanup()

	opts := &packages.Options{Env: append(os.Environ(), "GOPATH="+tmp)}
	initial, err := packages.Metadata(opts, "c", "subdir/d", "e")
	if err != nil {
		t.Fatal(err)
	}

	// Check graph topology.
	graph, all := importGraph(initial)
	wantGraph := `
  a
  b
* c
* e
  errors
* subdir/d
  unsafe
  b -> a
  b -> errors
  c -> b
  c -> unsafe
  e -> b
  e -> c
`[1:]

	if graph != wantGraph {
		t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
	}

	opts.Tests = true
	initial, err = packages.Metadata(opts, "c", "subdir/d", "e")
	if err != nil {
		t.Fatal(err)
	}

	// Check graph topology.
	graph, all = importGraph(initial)
	wantGraph = `
  a
  b
* c
* e
  errors
  math/bits
* subdir/d
  subdir/d [subdir/d.test]
* subdir/d.test
  subdir/d_test [subdir/d.test]
  unsafe
  b -> a
  b -> errors
  c -> b
  c -> unsafe
  e -> b
  e -> c
  subdir/d [subdir/d.test] -> math/bits
  subdir/d.test -> os (pruned)
  subdir/d.test -> subdir/d [subdir/d.test]
  subdir/d.test -> subdir/d_test [subdir/d.test]
  subdir/d.test -> testing (pruned)
  subdir/d.test -> testing/internal/testdeps (pruned)
  subdir/d_test [subdir/d.test] -> subdir/d [subdir/d.test]
`[1:]

	if graph != wantGraph {
		t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
	}

	// Check node information: kind, name, srcs.
	for _, test := range []struct {
		id       string
		wantName string
		wantKind string
		wantSrcs string
	}{
		{"a", "a", "package", "a.go"},
		{"b", "b", "package", "b.go"},
		{"c", "c", "package", "c.go"}, // c2.go is ignored
		{"e", "main", "command", "e.go e2.go"},
		{"errors", "errors", "package", "errors.go"},
		{"subdir/d", "d", "package", "d.go"},
		{"subdir/d.test", "main", "command", "0.go"},
		{"unsafe", "unsafe", "package", ""},
	} {
		p, ok := all[test.id]
		if !ok {
			t.Errorf("no package %s", test.id)
			continue
		}
		if p.Name != test.wantName {
			t.Errorf("%s.Name = %q, want %q", test.id, p.Name, test.wantName)
		}

		// kind
		var kind string
		if p.Name == "main" {
			kind += "command"
		} else {
			kind += "package"
		}
		if kind != test.wantKind {
			t.Errorf("%s.Kind = %q, want %q", test.id, kind, test.wantKind)
		}

		if srcs := strings.Join(srcs(p), " "); srcs != test.wantSrcs {
			t.Errorf("%s.Srcs = [%s], want [%s]", test.id, srcs, test.wantSrcs)
		}
	}

	// Test an ad-hoc package, analogous to "go run hello.go".
	if initial, err := packages.Metadata(opts, filepath.Join(tmp, "src/c/c.go")); len(initial) == 0 {
		t.Errorf("failed to obtain metadata for ad-hoc package: %s", err)
	} else {
		got := fmt.Sprintf("%s %s", initial[0].ID, srcs(initial[0]))
		if want := "command-line-arguments [c.go]"; got != want {
			t.Errorf("oops: got %s, want %s", got, want)
		}
	}

	// Wildcards
	// See StdlibTest for effective test of "std" wildcard.
	// TODO(adonovan): test "all" returns everything in the current module.
	{
		// "..." (subdirectory)
		initial, err = packages.Metadata(opts, "subdir/...")
		if err != nil {
			t.Fatal(err)
		}
		const want = "[subdir/d subdir/e subdir/d.test]"
		if fmt.Sprint(initial) != want {
			t.Errorf("for subdir/... wildcard, got %s, want %s", initial, want)
		}
	}
}

func TestOptionsDir(t *testing.T) {
	tmp, cleanup := makeTree(t, map[string]string{
		"src/a/a.go":   `package a; const Name = "a" `,
		"src/a/b/b.go": `package b; const Name = "a/b"`,
		"src/b/b.go":   `package b; const Name = "b"`,
	})
	defer cleanup()

	for _, test := range []struct {
		dir     string
		pattern string
		want    string // value of Name constant, or error
	}{
		{"", "a", `"a"`},
		{"", "b", `"b"`},
		{"", "./a", "packages not found"},
		{"", "./b", "packages not found"},
		{filepath.Join(tmp, "/src"), "a", `"a"`},
		{filepath.Join(tmp, "/src"), "b", `"b"`},
		{filepath.Join(tmp, "/src"), "./a", `"a"`},
		{filepath.Join(tmp, "/src"), "./b", `"b"`},
		{filepath.Join(tmp, "/src/a"), "a", `"a"`},
		{filepath.Join(tmp, "/src/a"), "b", `"b"`},
		{filepath.Join(tmp, "/src/a"), "./a", "packages not found"},
		{filepath.Join(tmp, "/src/a"), "./b", `"a/b"`},
	} {
		opts := &packages.Options{
			Dir: test.dir,
			Env: append(os.Environ(), "GOPATH="+tmp),
		}
		// Use TypeCheck to ensure that files can be opened.
		initial, err := packages.TypeCheck(opts, test.pattern)
		var got string
		if err != nil {
			got = err.Error()
		} else {
			got = constant(initial[0], "Name").Val().String()
		}
		if got != test.want {
			t.Errorf("dir %q, pattern %q: got %s, want %s",
				test.dir, test.pattern, got, test.want)
		}
	}

}

type errCollector struct {
	mu     sync.Mutex
	errors []error
}

func (ec *errCollector) add(err error) {
	ec.mu.Lock()
	ec.errors = append(ec.errors, err)
	ec.mu.Unlock()
}

func TestTypeCheckOK(t *testing.T) {
	tmp, cleanup := makeTree(t, map[string]string{
		"src/a/a.go": `package a; import "b"; const A = "a" + b.B`,
		"src/b/b.go": `package b; import "c"; const B = "b" + c.C`,
		"src/c/c.go": `package c; import "d"; const C = "c" + d.D`,
		"src/d/d.go": `package d; import "e"; const D = "d" + e.E`,
		"src/e/e.go": `package e; const E = "e"`,
	})
	defer cleanup()

	opts := &packages.Options{Env: append(os.Environ(), "GOPATH="+tmp), Error: func(error) {}}
	initial, err := packages.TypeCheck(opts, "a", "c")
	if err != nil {
		t.Fatal(err)
	}

	graph, all := importGraph(initial)
	wantGraph := `
* a
  b
* c
  d
  e
  a -> b
  b -> c
  c -> d
  d -> e
`[1:]
	if graph != wantGraph {
		t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
	}

	for _, test := range []struct {
		id        string
		wantType  bool
		wantFiles bool
	}{
		{"a", true, true},   // source package
		{"b", true, true},   // source package
		{"c", true, true},   // source package
		{"d", true, false},  // export data package
		{"e", false, false}, // no package
	} {
		p := all[test.id]
		if p == nil {
			t.Errorf("missing package: %s", test.id)
			continue
		}
		if (p.Type != nil) != test.wantType {
			if test.wantType {
				t.Errorf("missing types.Package for %s", p)
			} else {
				t.Errorf("unexpected types.Package for %s", p)
			}
		}
		if (p.Files != nil) != test.wantFiles {
			if test.wantFiles {
				t.Errorf("missing ast.Files for %s", p)
			} else {
				t.Errorf("unexpected ast.Files for for %s", p)
			}
		}
		if p.Errors != nil {
			t.Errorf("errors in package: %s: %s", p, p.Errors)
		}
	}

	// Check value of constant.
	aA := constant(all["a"], "A")
	if got, want := fmt.Sprintf("%v %v", aA, aA.Val()), `const a.A untyped string "abcde"`; got != want {
		t.Errorf("a.A: got %s, want %s", got, want)
	}
}

func TestTypeCheckError(t *testing.T) {
	// A type error in a lower-level package (e) prevents go list
	// from producing export data for all packages that depend on it
	// [a-e]. Export data is only required for package d, so package
	// c, which imports d, gets an error, and all packages above d
	// are IllTyped. Package e is not ill-typed, because the user
	// did not demand its type information (despite it actually
	// containing a type error).
	tmp, cleanup := makeTree(t, map[string]string{
		"src/a/a.go": `package a; import "b"; const A = "a" + b.B`,
		"src/b/b.go": `package b; import "c"; const B = "b" + c.C`,
		"src/c/c.go": `package c; import "d"; const C = "c" + d.D`,
		"src/d/d.go": `package d; import "e"; const D = "d" + e.E`,
		"src/e/e.go": `package e; const E = "e" + 1`, // type error
	})
	defer cleanup()

	opts := &packages.Options{Env: append(os.Environ(), "GOPATH="+tmp), Error: func(error) {}}
	initial, err := packages.TypeCheck(opts, "a", "c")
	if err != nil {
		t.Fatal(err)
	}

	all := packages.All(initial)

	for _, test := range []struct {
		id           string
		wantType     bool
		wantFiles    bool
		wantIllTyped bool
		wantErrs     []string
	}{
		{"a", true, true, true, nil},
		{"b", true, true, true, nil},
		{"c", true, true, true, []string{"could not import d (no export data file)"}},
		{"d", false, false, true, nil},  // missing export data
		{"e", false, false, false, nil}, // type info not requested (despite type error)
	} {
		p := all[test.id]
		if p == nil {
			t.Errorf("missing package: %s", test.id)
			continue
		}
		if (p.Type != nil) != test.wantType {
			if test.wantType {
				t.Errorf("missing types.Package for %s", test.id)
			} else {
				t.Errorf("unexpected types.Package for %s", test.id)
			}
		}
		if (p.Files != nil) != test.wantFiles {
			if test.wantFiles {
				t.Errorf("missing ast.Files for %s", test.id)
			} else {
				t.Errorf("unexpected ast.Files for for %s", test.id)
			}
		}
		if p.IllTyped != test.wantIllTyped {
			t.Errorf("IllTyped was %t for %s", p.IllTyped, test.id)
		}
		if errs := errorMessages(p.Errors); !reflect.DeepEqual(errs, test.wantErrs) {
			t.Errorf("in package %s, got errors %s, want %s", p, errs, test.wantErrs)
		}
	}

	// Check value of constant.
	aA := constant(all["a"], "A")
	if got, want := aA.String(), `const a.A invalid type`; got != want {
		t.Errorf("a.A: got %s, want %s", got, want)
	}
}

// This function tests use of the ParseFile hook to supply
// alternative file contents to the parser and type-checker.
func TestWholeProgramOverlay(t *testing.T) {
	type M = map[string]string

	tmp, cleanup := makeTree(t, M{
		"src/a/a.go": `package a; import "b"; const A = "a" + b.B`,
		"src/b/b.go": `package b; import "c"; const B = "b" + c.C`,
		"src/c/c.go": `package c; const C = "c"`,
		"src/d/d.go": `package d; const D = "d"`,
	})
	defer cleanup()

	for i, test := range []struct {
		overlay  M
		want     string // expected value of a.A
		wantErrs []string
	}{
		{nil, `"abc"`, nil}, // default
		{M{}, `"abc"`, nil}, // empty overlay
		{M{filepath.Join(tmp, "src/c/c.go"): `package c; const C = "C"`}, `"abC"`, nil},
		{M{filepath.Join(tmp, "src/b/b.go"): `package b; import "c"; const B = "B" + c.C`}, `"aBc"`, nil},
		{M{filepath.Join(tmp, "src/b/b.go"): `package b; import "d"; const B = "B" + d.D`}, `unknown`,
			[]string{`could not import d (no metadata for d)`}},
	} {
		var parseFile func(fset *token.FileSet, filename string) (*ast.File, error)
		if test.overlay != nil {
			parseFile = func(fset *token.FileSet, filename string) (*ast.File, error) {
				var src interface{}
				if content, ok := test.overlay[filename]; ok {
					src = content
				}
				const mode = parser.AllErrors | parser.ParseComments
				return parser.ParseFile(fset, filename, src, mode)
			}
		}
		var errs errCollector
		opts := &packages.Options{
			Env:       append(os.Environ(), "GOPATH="+tmp),
			Error:     errs.add,
			ParseFile: parseFile,
		}
		initial, err := packages.WholeProgram(opts, "a")
		if err != nil {
			t.Error(err)
			continue
		}

		// Check value of a.A.
		a := initial[0]
		got := constant(a, "A").Val().String()
		if got != test.want {
			t.Errorf("%d. a.A: got %s, want %s", i, got, test.want)
		}

		if errs := errorMessages(errs.errors); !reflect.DeepEqual(errs, test.wantErrs) {
			t.Errorf("%d. got errors %s, want %s", i, errs, test.wantErrs)
		}
	}
}

func TestWholeProgramImportErrors(t *testing.T) {
	tmp, cleanup := makeTree(t, map[string]string{
		"src/unicycle/unicycle.go": `package unicycle; import _ "unicycle"`,
		"src/bicycle1/bicycle1.go": `package bicycle1; import _ "bicycle2"`,
		"src/bicycle2/bicycle2.go": `package bicycle2; import _ "bicycle1"`,
		"src/bad/bad.go":           `not a package declaration`,
		"src/root/root.go": `package root
import (
	_ "bicycle1"
	_ "unicycle"
	_ "nonesuch"
	_ "empty"
	_ "bad"
)`,
	})
	defer cleanup()

	os.Mkdir(filepath.Join(tmp, "src/empty"), 0777) // create an existing but empty package

	var errs2 errCollector
	opts := &packages.Options{
		Env:   append(os.Environ(), "GOPATH="+tmp),
		Error: errs2.add,
	}
	initial, err := packages.WholeProgram(opts, "root")
	if err != nil {
		t.Fatal(err)
	}

	// Cycle-forming edges are removed from the graph:
	// 	bicycle2 -> bicycle1
	//      unicycle -> unicycle
	graph, all := importGraph(initial)
	wantGraph := `
  bicycle1
  bicycle2
* root
  unicycle
  bicycle1 -> bicycle2
  root -> bicycle1
  root -> unicycle
`[1:]
	if graph != wantGraph {
		t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
	}
	for _, test := range []struct {
		id       string
		wantErrs []string
	}{
		{"bicycle1", nil},
		{"bicycle2", []string{
			"could not import bicycle1 (import cycle: [root bicycle1 bicycle2])",
		}},
		{"unicycle", []string{
			"could not import unicycle (import cycle: [root unicycle])",
		}},
		{"root", []string{
			`could not import bad (missing package: "bad")`,
			`could not import empty (missing package: "empty")`,
			`could not import nonesuch (missing package: "nonesuch")`,
		}},
	} {
		p := all[test.id]
		if p == nil {
			t.Errorf("missing package: %s", test.id)
			continue
		}
		if p.Type == nil {
			t.Errorf("missing types.Package for %s", test.id)
		}
		if p.Files == nil {
			t.Errorf("missing ast.Files for %s", test.id)
		}
		if !p.IllTyped {
			t.Errorf("IllTyped was false for %s", test.id)
		}
		if errs := errorMessages(p.Errors); !reflect.DeepEqual(errs, test.wantErrs) {
			t.Errorf("in package %s, got errors %s, want %s", p, errs, test.wantErrs)
		}
	}
}

func errorMessages(errors []error) []string {
	var msgs []string
	for _, err := range errors {
		msg := err.Error()
		// Strip off /tmp filename.
		if i := strings.Index(msg, ": "); i >= 0 {
			msg = msg[i+len(": "):]
		}
		msgs = append(msgs, msg)
	}
	sort.Strings(msgs)
	return msgs
}

func srcs(p *packages.Package) (basenames []string) {
	for i, src := range p.Srcs {
		if strings.Contains(src, ".cache/go-build") {
			src = fmt.Sprintf("%d.go", i) // make cache names predictable
		} else {
			src = filepath.Base(src)
		}
		basenames = append(basenames, src)
	}
	return basenames
}

// importGraph returns the import graph as a user-friendly string,
// and a map containing all packages keyed by ID.
func importGraph(initial []*packages.Package) (string, map[string]*packages.Package) {
	out := new(bytes.Buffer)

	initialSet := make(map[*packages.Package]bool)
	for _, p := range initial {
		initialSet[p] = true
	}

	// We can't use packages.All because
	// we need to prune the traversal.
	var nodes, edges []string
	res := make(map[string]*packages.Package)
	seen := make(map[*packages.Package]bool)
	var visit func(p *packages.Package)
	visit = func(p *packages.Package) {
		if !seen[p] {
			seen[p] = true
			if res[p.ID] != nil {
				panic("duplicate ID: " + p.ID)
			}
			res[p.ID] = p

			star := ' ' // mark initial packages with a star
			if initialSet[p] {
				star = '*'
			}
			nodes = append(nodes, fmt.Sprintf("%c %s", star, p.ID))

			// To avoid a lot of noise,
			// we prune uninteresting dependencies of testmain packages,
			// which we identify by this import:
			isTestMain := p.Imports["testing/internal/testdeps"] != nil

			for _, imp := range p.Imports {
				if isTestMain {
					switch imp.ID {
					case "os", "testing", "testing/internal/testdeps":
						edges = append(edges, fmt.Sprintf("%s -> %s (pruned)", p, imp))
						continue
					}
				}
				edges = append(edges, fmt.Sprintf("%s -> %s", p, imp))
				visit(imp)
			}
		}
	}
	for _, p := range initial {
		visit(p)
	}

	// Sort, ignoring leading optional star prefix.
	sort.Slice(nodes, func(i, j int) bool { return nodes[i][2:] < nodes[j][2:] })
	for _, node := range nodes {
		fmt.Fprintf(out, "%s\n", node)
	}

	sort.Strings(edges)
	for _, edge := range edges {
		fmt.Fprintf(out, "  %s\n", edge)
	}

	return out.String(), res
}

const skipCleanup = false // for debugging; don't commit 'true'!

// makeTree creates a new temporary directory containing the specified
// file tree, and chdirs to it. Call the cleanup function to restore the
// cwd and delete the tree.
func makeTree(t *testing.T, tree map[string]string) (dir string, cleanup func()) {
	dir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatal(err)
	}

	cleanup = func() {
		if skipCleanup {
			t.Logf("Skipping cleanup of temp dir: %s", dir)
			return
		}
		os.RemoveAll(dir) // ignore errors
	}

	for name, content := range tree {
		name := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(name), 0777); err != nil {
			cleanup()
			t.Fatal(err)
		}
		if err := ioutil.WriteFile(name, []byte(content), 0666); err != nil {
			cleanup()
			t.Fatal(err)
		}
	}
	return dir, cleanup
}

func constant(p *packages.Package, name string) *types.Const {
	return p.Type.Scope().Lookup(name).(*types.Const)
}
