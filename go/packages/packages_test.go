// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packages_test

import (
	"bytes"
	"encoding/json"
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
	"testing"

	"golang.org/x/tools/go/packages"
)

func init() {
	// Insulate the tests from the users' environment.
	os.Setenv("GOPACKAGESDRIVER", "off")
}

// TODO(matloob): remove this once Go 1.12 is released as we will end support
// for versions of go list before Go 1.10.4.
var usesOldGolist = false

// TODO(adonovan): more test cases to write:
//
// - When the tests fail, make them print a 'cd & load' command
//   that will allow the maintainer to interact with the failing scenario.
// - errors in go-list metadata
// - a foo.test package that cannot be built for some reason (e.g.
//   import error) will result in a JSON blob with no name and a
//   nonexistent testmain file in GoFiles. Test that we handle this
//   gracefully.
// - test more Flags.
//
// LoadSyntax & LoadAllSyntax modes:
//   - Fset may be user-supplied or not.
//   - Packages.Info is correctly set.
//   - typechecker configuration is honored
//   - import cycles are gracefully handled in type checker.
//   - test typechecking of generated test main and cgo.

// The zero-value of Config has LoadFiles mode.
func TestLoadZeroConfig(t *testing.T) {
	initial, err := packages.Load(nil, "hash")
	if err != nil {
		t.Fatal(err)
	}
	if len(initial) != 1 {
		t.Fatalf("got %s, want [hash]", initial)
	}
	hash := initial[0]
	// Even though the hash package has imports,
	// they are not reported.
	got := fmt.Sprintf("name=%s srcs=%v imports=%v", hash.Name, srcs(hash), hash.Imports)
	want := "name=hash srcs=[hash.go] imports=map[]"
	if got != want {
		t.Fatalf("got %s, want %s", got, want)
	}
}

func TestLoadImportsGraph(t *testing.T) {
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

	cfg := &packages.Config{
		Mode: packages.LoadImports,
		Env:  append(os.Environ(), "GOPATH="+tmp, "GO111MODULE=off"),
	}
	initial, err := packages.Load(cfg, "c", "subdir/d", "e")
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

	cfg.Tests = true
	initial, err = packages.Load(cfg, "c", "subdir/d", "e")
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
* subdir/d [subdir/d.test]
* subdir/d.test
* subdir/d_test [subdir/d.test]
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
	if initial, err := packages.Load(cfg, filepath.Join(tmp, "src/c/c.go")); len(initial) == 0 {
		t.Errorf("failed to obtain metadata for ad-hoc package: %s", err)
	} else {
		got := fmt.Sprintf("%s %s", initial[0].ID, srcs(initial[0]))
		if want := "command-line-arguments [c.go]"; got != want && !usesOldGolist {
			t.Errorf("oops: got %s, want %s", got, want)
		}
	}

	// Wildcards
	// See StdlibTest for effective test of "std" wildcard.
	// TODO(adonovan): test "all" returns everything in the current module.
	{
		// "..." (subdirectory)
		initial, err = packages.Load(cfg, "subdir/...")
		if err != nil {
			t.Fatal(err)
		}
		graph, all = importGraph(initial)
		wantGraph = `
  math/bits
* subdir/d
* subdir/d [subdir/d.test]
* subdir/d.test
* subdir/d_test [subdir/d.test]
* subdir/e
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
	}
}

func TestLoadImportsTestVariants(t *testing.T) {
	if usesOldGolist {
		t.Skip("not yet supported in pre-Go 1.10.4 golist fallback implementation")
	}

	tmp, cleanup := makeTree(t, map[string]string{
		"src/a/a.go":       `package a; import _ "b"`,
		"src/b/b.go":       `package b`,
		"src/b/b_test.go":  `package b`,
		"src/b/bx_test.go": `package b_test; import _ "a"`,
	})
	defer cleanup()

	cfg := &packages.Config{
		Mode:  packages.LoadImports,
		Env:   append(os.Environ(), "GOPATH="+tmp, "GO111MODULE=off"),
		Tests: true,
	}
	initial, err := packages.Load(cfg, "a", "b")
	if err != nil {
		t.Fatal(err)
	}

	// Check graph topology.
	graph, _ := importGraph(initial)
	wantGraph := `
* a
  a [b.test]
* b
* b [b.test]
* b.test
* b_test [b.test]
  a -> b
  a [b.test] -> b [b.test]
  b.test -> b [b.test]
  b.test -> b_test [b.test]
  b.test -> os (pruned)
  b.test -> testing (pruned)
  b.test -> testing/internal/testdeps (pruned)
  b_test [b.test] -> a [b.test]
`[1:]

	if graph != wantGraph {
		t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
	}
}

func TestLoadImportsC(t *testing.T) {
	// This test checks that when a package depends on the
	// test variant of "syscall", "unsafe", or "runtime/cgo", that dependency
	// is not removed when those packages are added when it imports "C".
	//
	// For this test to work, the external test of syscall must have a dependency
	// on net, and net must import "syscall" and "C".
	if runtime.GOOS == "windows" {
		t.Skipf("skipping on windows; packages on windows do not satisfy conditions for test.")
	}
	if runtime.GOOS == "plan9" {
		// See https://github.com/golang/go/issues/27100.
		t.Skip(`skipping on plan9; for some reason "net [syscall.test]" is not loaded`)
	}
	if usesOldGolist {
		t.Skip("not yet supported in pre-Go 1.10.4 golist fallback implementation")
	}

	cfg := &packages.Config{
		Mode:  packages.LoadImports,
		Tests: true,
	}
	initial, err := packages.Load(cfg, "syscall", "net")
	if err != nil {
		t.Fatalf("failed to load imports: %v", err)
	}

	_, all := importGraph(initial)

	for _, test := range []struct {
		pattern    string
		wantImport string // an import to check for
	}{
		{"net", "syscall:syscall"},
		{"net [syscall.test]", "syscall:syscall [syscall.test]"},
		{"syscall_test [syscall.test]", "net:net [syscall.test]"},
	} {
		// Test the import paths.
		pkg := all[test.pattern]
		if pkg == nil {
			t.Errorf("package %q not loaded", test.pattern)
			continue
		}
		if imports := strings.Join(imports(pkg), " "); !strings.Contains(imports, test.wantImport) {
			t.Errorf("package %q: got \n%s, \nwant to have %s", test.pattern, imports, test.wantImport)
		}
	}
}

func TestVendorImports(t *testing.T) {
	tmp, cleanup := makeTree(t, map[string]string{
		"src/a/a.go":          `package a; import _ "b"; import _ "c";`,
		"src/a/vendor/b/b.go": `package b; import _ "c"`,
		"src/c/c.go":          `package c; import _ "b"`,
		"src/c/vendor/b/b.go": `package b`,
	})
	defer cleanup()

	cfg := &packages.Config{
		Mode: packages.LoadImports,
		Env:  append(os.Environ(), "GOPATH="+tmp, "GO111MODULE=off"),
	}
	initial, err := packages.Load(cfg, "a", "c")
	if err != nil {
		t.Fatal(err)
	}

	graph, all := importGraph(initial)
	wantGraph := `
* a
  a/vendor/b
* c
  c/vendor/b
  a -> a/vendor/b
  a -> c
  a/vendor/b -> c
  c -> c/vendor/b
`[1:]
	if graph != wantGraph {
		t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
	}

	for _, test := range []struct {
		pattern     string
		wantImports string
	}{
		{"a", "b:a/vendor/b c:c"},
		{"c", "b:c/vendor/b"},
		{"a/vendor/b", "c:c"},
		{"c/vendor/b", ""},
	} {
		// Test the import paths.
		pkg := all[test.pattern]
		if imports := strings.Join(imports(pkg), " "); imports != test.wantImports {
			t.Errorf("package %q: got %s, want %s", test.pattern, imports, test.wantImports)
		}
	}
}

func imports(p *packages.Package) []string {
	if p == nil {
		return nil
	}
	keys := make([]string, 0, len(p.Imports))
	for k, v := range p.Imports {
		keys = append(keys, fmt.Sprintf("%s:%s", k, v.ID))
	}
	sort.Strings(keys)
	return keys
}

func TestConfigDir(t *testing.T) {
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
		cfg := &packages.Config{
			Mode: packages.LoadSyntax, // Use LoadSyntax to ensure that files can be opened.
			Dir:  test.dir,
			Env:  append(os.Environ(), "GOPATH="+tmp, "GO111MODULE=off"),
		}

		initial, err := packages.Load(cfg, test.pattern)
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

func TestConfigFlags(t *testing.T) {
	// Test satisfying +build line tags, with -tags flag.
	tmp, cleanup := makeTree(t, map[string]string{
		// package a
		"src/a/a.go": `package a; import _ "a/b"`,
		"src/a/b.go": `// +build tag

package a`,
		"src/a/c.go": `// +build tag tag2

package a`,
		"src/a/d.go": `// +build tag,tag2

package a`,
		// package a/b
		"src/a/b/a.go": `package b`,
		"src/a/b/b.go": `// +build tag

package b`,
	})
	defer cleanup()

	for _, test := range []struct {
		pattern        string
		tags           []string
		wantSrcs       string
		wantImportSrcs string
	}{
		{`a`, []string{}, "a.go", "a.go"},
		{`a`, []string{`-tags=tag`}, "a.go b.go c.go", "a.go b.go"},
		{`a`, []string{`-tags=tag2`}, "a.go c.go", "a.go"},
		{`a`, []string{`-tags=tag tag2`}, "a.go b.go c.go d.go", "a.go b.go"},
	} {
		cfg := &packages.Config{
			Mode:       packages.LoadImports,
			BuildFlags: test.tags,
			Env:        append(os.Environ(), "GOPATH="+tmp, "GO111MODULE=off"),
		}

		initial, err := packages.Load(cfg, test.pattern)
		if err != nil {
			t.Error(err)
			continue
		}
		if len(initial) != 1 {
			t.Errorf("test tags %v: pattern %s, expected 1 package, got %d packages.", test.tags, test.pattern, len(initial))
			continue
		}
		pkg := initial[0]
		if srcs := strings.Join(srcs(pkg), " "); srcs != test.wantSrcs {
			t.Errorf("test tags %v: srcs of package %s = [%s], want [%s]", test.tags, test.pattern, srcs, test.wantSrcs)
		}
		for path, ipkg := range pkg.Imports {
			if srcs := strings.Join(srcs(ipkg), " "); srcs != test.wantImportSrcs {
				t.Errorf("build tags %v: srcs of imported package %s = [%s], want [%s]", test.tags, path, srcs, test.wantImportSrcs)
			}
		}

	}
}

func TestLoadTypes(t *testing.T) {
	// In LoadTypes and LoadSyntax modes, the compiler will
	// fail to generate an export data file for c, because it has
	// a type error.  The loader should fall back loading a and c
	// from source, but use the export data for b.

	tmp, cleanup := makeTree(t, map[string]string{
		"src/a/a.go": `package a; import "b"; import "c"; const A = "a" + b.B + c.C`,
		"src/b/b.go": `package b; const B = "b"`,
		"src/c/c.go": `package c; const C = "c" + 1`,
	})
	defer cleanup()

	cfg := &packages.Config{
		Mode: packages.LoadTypes,
		Env:  append(os.Environ(), "GOPATH="+tmp, "GO111MODULE=off"),
	}
	initial, err := packages.Load(cfg, "a")
	if err != nil {
		t.Fatal(err)
	}

	graph, all := importGraph(initial)
	wantGraph := `
* a
  b
  c
  a -> b
  a -> c
`[1:]
	if graph != wantGraph {
		t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
	}

	for _, test := range []struct {
		id         string
		wantSyntax bool
	}{
		{"a", true},  // need src, no export data for c
		{"b", false}, // use export data
		{"c", true},  // need src, no export data for c
	} {
		if usesOldGolist && !test.wantSyntax {
			// legacy go list always upgrades to LoadAllSyntax, syntax will be filled in.
			// still check that types information is complete.
			test.wantSyntax = true
		}
		p := all[test.id]
		if p == nil {
			t.Errorf("missing package: %s", test.id)
			continue
		}
		if p.Types == nil {
			t.Errorf("missing types.Package for %s", p)
			continue
		} else if !p.Types.Complete() {
			t.Errorf("incomplete types.Package for %s", p)
		}
		if (p.Syntax != nil) != test.wantSyntax {
			if test.wantSyntax {
				t.Errorf("missing ast.Files for %s", p)
			} else {
				t.Errorf("unexpected ast.Files for for %s", p)
			}
		}
	}
}

func TestLoadSyntaxOK(t *testing.T) {
	tmp, cleanup := makeTree(t, map[string]string{
		"src/a/a.go": `package a; import "b"; const A = "a" + b.B`,
		"src/b/b.go": `package b; import "c"; const B = "b" + c.C`,
		"src/c/c.go": `package c; import "d"; const C = "c" + d.D`,
		"src/d/d.go": `package d; import "e"; const D = "d" + e.E`,
		"src/e/e.go": `package e; import "f"; const E = "e" + f.F`,
		"src/f/f.go": `package f; const F = "f"`,
	})
	defer cleanup()

	cfg := &packages.Config{
		Mode: packages.LoadSyntax,
		Env:  append(os.Environ(), "GOPATH="+tmp, "GO111MODULE=off"),
	}
	initial, err := packages.Load(cfg, "a", "c")
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
  f
  a -> b
  b -> c
  c -> d
  d -> e
  e -> f
`[1:]
	if graph != wantGraph {
		t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
	}

	for _, test := range []struct {
		id           string
		wantSyntax   bool
		wantComplete bool
	}{
		{"a", true, true},   // source package
		{"b", true, true},   // source package
		{"c", true, true},   // source package
		{"d", false, true},  // export data package
		{"e", false, false}, // export data package
		{"f", false, false}, // export data package
	} {
		// TODO(matloob): The legacy go list based support loads
		// everything from source because it doesn't do a build
		// and the .a files don't exist.
		// Can we simulate its existence?
		if usesOldGolist {
			test.wantComplete = true
			test.wantSyntax = true
		}
		p := all[test.id]
		if p == nil {
			t.Errorf("missing package: %s", test.id)
			continue
		}
		if p.Types == nil {
			t.Errorf("missing types.Package for %s", p)
			continue
		} else if p.Types.Complete() != test.wantComplete {
			if test.wantComplete {
				t.Errorf("incomplete types.Package for %s", p)
			} else {
				t.Errorf("unexpected complete types.Package for %s", p)
			}
		}
		if (p.Syntax != nil) != test.wantSyntax {
			if test.wantSyntax {
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
	if got, want := fmt.Sprintf("%v %v", aA, aA.Val()), `const a.A untyped string "abcdef"`; got != want {
		t.Errorf("a.A: got %s, want %s", got, want)
	}
}

func TestLoadDiamondTypes(t *testing.T) {
	// We make a diamond dependency and check the type d.D is the same through both paths
	tmp, cleanup := makeTree(t, map[string]string{
		"src/a/a.go": `package a; import ("b"; "c"); var _ = b.B == c.C`,
		"src/b/b.go": `package b; import "d"; var B d.D`,
		"src/c/c.go": `package c; import "d"; var C d.D`,
		"src/d/d.go": `package d; type D int`,
	})
	defer cleanup()

	cfg := &packages.Config{
		Mode: packages.LoadSyntax,
		Env:  append(os.Environ(), "GOPATH="+tmp, "GO111MODULE=off"),
	}
	initial, err := packages.Load(cfg, "a")
	if err != nil {
		t.Fatal(err)
	}
	packages.Visit(initial, nil, func(pkg *packages.Package) {
		for _, err := range pkg.Errors {
			t.Errorf("package %s: %v", pkg.ID, err)
		}
	})

	graph, _ := importGraph(initial)
	wantGraph := `
* a
  b
  c
  d
  a -> b
  a -> c
  b -> d
  c -> d
`[1:]
	if graph != wantGraph {
		t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
	}
}

func TestLoadSyntaxError(t *testing.T) {
	// A type error in a lower-level package (e) prevents go list
	// from producing export data for all packages that depend on it
	// [a-e]. Only f should be loaded from export data, and the rest
	// should be IllTyped.
	tmp, cleanup := makeTree(t, map[string]string{
		"src/a/a.go": `package a; import "b"; const A = "a" + b.B`,
		"src/b/b.go": `package b; import "c"; const B = "b" + c.C`,
		"src/c/c.go": `package c; import "d"; const C = "c" + d.D`,
		"src/d/d.go": `package d; import "e"; const D = "d" + e.E`,
		"src/e/e.go": `package e; import "f"; const E = "e" + f.F + 1`, // type error
		"src/f/f.go": `package f; const F = "f"`,
	})
	defer cleanup()

	cfg := &packages.Config{
		Mode: packages.LoadSyntax,
		Env:  append(os.Environ(), "GOPATH="+tmp, "GO111MODULE=off"),
	}
	initial, err := packages.Load(cfg, "a", "c")
	if err != nil {
		t.Fatal(err)
	}

	all := make(map[string]*packages.Package)
	packages.Visit(initial, nil, func(p *packages.Package) {
		all[p.ID] = p
	})

	for _, test := range []struct {
		id           string
		wantSyntax   bool
		wantIllTyped bool
	}{
		{"a", true, true},
		{"b", true, true},
		{"c", true, true},
		{"d", true, true},
		{"e", true, true},
		{"f", false, false},
	} {
		if usesOldGolist && !test.wantSyntax {
			// legacy go list always upgrades to LoadAllSyntax, syntax will be filled in.
			test.wantSyntax = true
		}
		p := all[test.id]
		if p == nil {
			t.Errorf("missing package: %s", test.id)
			continue
		}
		if p.Types == nil {
			t.Errorf("missing types.Package for %s", p)
			continue
		} else if !p.Types.Complete() {
			t.Errorf("incomplete types.Package for %s", p)
		}
		if (p.Syntax != nil) != test.wantSyntax {
			if test.wantSyntax {
				t.Errorf("missing ast.Files for %s", test.id)
			} else {
				t.Errorf("unexpected ast.Files for for %s", test.id)
			}
		}
		if p.IllTyped != test.wantIllTyped {
			t.Errorf("IllTyped was %t for %s", p.IllTyped, test.id)
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
func TestLoadAllSyntaxOverlay(t *testing.T) {
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
		cfg := &packages.Config{
			Mode:      packages.LoadAllSyntax,
			Env:       append(os.Environ(), "GOPATH="+tmp, "GO111MODULE=off"),
			ParseFile: parseFile,
		}
		initial, err := packages.Load(cfg, "a")
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

		// Check errors.
		var errors []packages.Error
		packages.Visit(initial, nil, func(pkg *packages.Package) {
			errors = append(errors, pkg.Errors...)
		})
		if errs := errorMessages(errors); !reflect.DeepEqual(errs, test.wantErrs) {
			t.Errorf("%d. got errors %s, want %s", i, errs, test.wantErrs)
		}
	}
}

func TestLoadAllSyntaxImportErrors(t *testing.T) {
	// TODO(matloob): Remove this once go list -e -compiled is fixed. See golang.org/issue/26755
	t.Skip("go list -compiled -e fails with non-zero exit status for empty packages")

	if usesOldGolist {
		t.Skip("not yet supported in pre-Go 1.10.4 golist fallback implementation")
	}

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

	cfg := &packages.Config{
		Mode: packages.LoadAllSyntax,
		Env:  append(os.Environ(), "GOPATH="+tmp, "GO111MODULE=off"),
	}
	initial, err := packages.Load(cfg, "root")
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
		if p.Types == nil {
			t.Errorf("missing types.Package for %s", test.id)
		}
		if p.Syntax == nil {
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

func TestAbsoluteFilenames(t *testing.T) {
	tmp, cleanup := makeTree(t, map[string]string{
		"src/a/a.go":          `package a; const A = 1`,
		"src/b/b.go":          `package b; import ("a"; _ "errors"); var B = a.A`,
		"src/b/vendor/a/a.go": `package a; const A = 1`,
		"src/c/c.go":          `package c; import (_ "b"; _ "unsafe")`,
		"src/c/c2.go":         "// +build ignore\n\n" + `package c; import _ "fmt"`,
		"src/subdir/d/d.go":   `package d`,
		"src/subdir/e/d.go":   `package e`,
		"src/e/e.go":          `package main; import _ "b"`,
		"src/e/e2.go":         `package main; import _ "c"`,
		"src/f/f.go":          `package f`,
		"src/f/f.s":           ``,
	})
	defer cleanup()

	checkFile := func(filename string) {
		if !filepath.IsAbs(filename) {
			t.Errorf("filename is not absolute: %s", filename)
		}
		if _, err := os.Stat(filename); err != nil {
			t.Errorf("stat error, %s: %v", filename, err)
		}
	}

	for _, test := range []struct {
		pattern string
		want    string
	}{
		// Import paths
		{"a", "a.go"},
		{"b/vendor/a", "a.go"},
		{"b", "b.go"},
		{"c", "c.go"},
		{"subdir/d", "d.go"},
		{"subdir/e", "d.go"},
		{"e", "e.go e2.go"},
		{"f", "f.go f.s"},
		// Relative paths
		{"./a", "a.go"},
		{"./b/vendor/a", "a.go"},
		{"./b", "b.go"},
		{"./c", "c.go"},
		{"./subdir/d", "d.go"},
		{"./subdir/e", "d.go"},
		{"./e", "e.go e2.go"},
		{"./f", "f.go f.s"},
	} {
		cfg := &packages.Config{
			Mode: packages.LoadFiles,
			Dir:  filepath.Join(tmp, "src"),
			Env:  append(os.Environ(), "GOPATH="+tmp, "GO111MODULE=off"),
		}
		pkgs, err := packages.Load(cfg, test.pattern)
		if err != nil {
			t.Errorf("pattern %s: %v", test.pattern, err)
			continue
		}

		if got := strings.Join(srcs(pkgs[0]), " "); got != test.want {
			t.Errorf("in package %s, got %s, want %s", test.pattern, got, test.want)
		}

		// Test that files in all packages exist and are absolute paths.
		_, all := importGraph(pkgs)
		for _, pkg := range all {
			for _, filename := range pkg.GoFiles {
				checkFile(filename)
			}
			for _, filename := range pkg.OtherFiles {
				checkFile(filename)
			}
		}
	}
}

func TestContains(t *testing.T) {
	tmp, cleanup := makeTree(t, map[string]string{
		"src/a/a.go": `package a; import "b"`,
		"src/b/b.go": `package b; import "c"`,
		"src/c/c.go": `package c`,
	})
	defer cleanup()

	cfg := &packages.Config{
		Mode: packages.LoadImports,
		Dir:  tmp,
		Env:  append(os.Environ(), "GOPATH="+tmp, "GO111MODULE=off"),
	}
	initial, err := packages.Load(cfg, "contains:src/b/b.go")
	if err != nil {
		t.Fatal(err)
	}

	graph, _ := importGraph(initial)
	wantGraph := `
* b
  c
  b -> c
`[1:]
	if graph != wantGraph {
		t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
	}
}

// TestContains_FallbackSticks ensures that when there are both contains and non-contains queries
// the decision whether to fallback to the pre-1.11 go list sticks across both sets of calls to
// go list.
func TestContains_FallbackSticks(t *testing.T) {
	tmp, cleanup := makeTree(t, map[string]string{
		"src/a/a.go": `package a; import "b"`,
		"src/b/b.go": `package b; import "c"`,
		"src/c/c.go": `package c`,
	})
	defer cleanup()

	cfg := &packages.Config{
		Mode: packages.LoadImports,
		Dir:  tmp,
		Env:  append(os.Environ(), "GOPATH="+tmp, "GO111MODULE=off"),
	}
	initial, err := packages.Load(cfg, "a", "contains:src/b/b.go")
	if err != nil {
		t.Fatal(err)
	}

	graph, _ := importGraph(initial)
	wantGraph := `
* a
* b
  c
  a -> b
  b -> c
`[1:]
	if graph != wantGraph {
		t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
	}
}

func TestJSON(t *testing.T) {
	//TODO: add in some errors
	tmp, cleanup := makeTree(t, map[string]string{
		"src/a/a.go": `package a; const A = 1`,
		"src/b/b.go": `package b; import "a"; var B = a.A`,
		"src/c/c.go": `package c; import "b" ; var C = b.B`,
		"src/d/d.go": `package d; import "b" ; var D = b.B`,
	})
	defer cleanup()

	cfg := &packages.Config{
		Mode: packages.LoadImports,
		Env:  append(os.Environ(), "GOPATH="+tmp, "GO111MODULE=off"),
	}
	initial, err := packages.Load(cfg, "c", "d")
	if err != nil {
		t.Fatal(err)
	}

	// Visit and print all packages.
	buf := &bytes.Buffer{}
	enc := json.NewEncoder(buf)
	enc.SetIndent("", "\t")
	packages.Visit(initial, nil, func(pkg *packages.Package) {
		// trim the source lists for stable results
		pkg.GoFiles = cleanPaths(pkg.GoFiles)
		pkg.CompiledGoFiles = cleanPaths(pkg.CompiledGoFiles)
		pkg.OtherFiles = cleanPaths(pkg.OtherFiles)
		if err := enc.Encode(pkg); err != nil {
			t.Fatal(err)
		}
	})

	wantJSON := `
{
	"ID": "a",
	"Name": "a",
	"PkgPath": "a",
	"GoFiles": [
		"a.go"
	],
	"CompiledGoFiles": [
		"a.go"
	]
}
{
	"ID": "b",
	"Name": "b",
	"PkgPath": "b",
	"GoFiles": [
		"b.go"
	],
	"CompiledGoFiles": [
		"b.go"
	],
	"Imports": {
		"a": "a"
	}
}
{
	"ID": "c",
	"Name": "c",
	"PkgPath": "c",
	"GoFiles": [
		"c.go"
	],
	"CompiledGoFiles": [
		"c.go"
	],
	"Imports": {
		"b": "b"
	}
}
{
	"ID": "d",
	"Name": "d",
	"PkgPath": "d",
	"GoFiles": [
		"d.go"
	],
	"CompiledGoFiles": [
		"d.go"
	],
	"Imports": {
		"b": "b"
	}
}
`[1:]

	if buf.String() != wantJSON {
		t.Errorf("wrong JSON: got <<%s>>, want <<%s>>", buf.String(), wantJSON)
	}
	// now decode it again
	var decoded []*packages.Package
	dec := json.NewDecoder(buf)
	for dec.More() {
		p := new(packages.Package)
		if err := dec.Decode(p); err != nil {
			t.Fatal(err)
		}
		decoded = append(decoded, p)
	}
	if len(decoded) != 4 {
		t.Fatalf("got %d packages, want 4", len(decoded))
	}
	for i, want := range []*packages.Package{{
		ID:   "a",
		Name: "a",
	}, {
		ID:   "b",
		Name: "b",
		Imports: map[string]*packages.Package{
			"a": &packages.Package{ID: "a"},
		},
	}, {
		ID:   "c",
		Name: "c",
		Imports: map[string]*packages.Package{
			"b": &packages.Package{ID: "b"},
		},
	}, {
		ID:   "d",
		Name: "d",
		Imports: map[string]*packages.Package{
			"b": &packages.Package{ID: "b"},
		},
	}} {
		got := decoded[i]
		if got.ID != want.ID {
			t.Errorf("Package %d has ID %q want %q", i, got.ID, want.ID)
		}
		if got.Name != want.Name {
			t.Errorf("Package %q has Name %q want %q", got.ID, got.Name, want.Name)
		}
		if len(got.Imports) != len(want.Imports) {
			t.Errorf("Package %q has %d imports want %d", got.ID, len(got.Imports), len(want.Imports))
			continue
		}
		for path, ipkg := range got.Imports {
			if want.Imports[path] == nil {
				t.Errorf("Package %q has unexpected import %q", got.ID, path)
				continue
			}
			if want.Imports[path].ID != ipkg.ID {
				t.Errorf("Package %q import %q is %q want %q", got.ID, path, ipkg.ID, want.Imports[path].ID)
			}
		}
	}
}

func TestConfigDefaultEnv(t *testing.T) {
	if runtime.GOOS == "windows" {
		// TODO(jayconrod): write an equivalent batch script for windows.
		// Hint: "type" can be used to read a file to stdout.
		t.Skip("test requires sh")
	}
	tmp, cleanup := makeTree(t, map[string]string{
		"bin/gopackagesdriver": `#!/bin/sh

cat - <<'EOF'
{
  "Roots": ["gopackagesdriver"],
  "Packages": [{"ID": "gopackagesdriver", "Name": "gopackagesdriver"}]
}
EOF
`,
		"src/golist/golist.go": "package golist",
	})
	defer cleanup()
	if err := os.Chmod(filepath.Join(tmp, "bin", "gopackagesdriver"), 0755); err != nil {
		t.Fatal(err)
	}

	path, ok := os.LookupEnv("PATH")
	var pathWithDriver string
	if ok {
		pathWithDriver = filepath.Join(tmp, "bin") + string(os.PathListSeparator) + path
	} else {
		pathWithDriver = filepath.Join(tmp, "bin")
	}

	for _, test := range []struct {
		desc    string
		env     []string
		wantIDs string
	}{
		{
			desc:    "driver_off",
			env:     []string{"PATH", pathWithDriver, "GOPATH", tmp, "GOPACKAGESDRIVER", "off"},
			wantIDs: "[golist]",
		}, {
			desc:    "driver_unset",
			env:     []string{"PATH", pathWithDriver, "GOPATH", "", "GOPACKAGESDRIVER", ""},
			wantIDs: "[gopackagesdriver]",
		}, {
			desc:    "driver_set",
			env:     []string{"GOPACKAGESDRIVER", filepath.Join(tmp, "bin", "gopackagesdriver")},
			wantIDs: "[gopackagesdriver]",
		},
	} {
		t.Run(test.desc, func(t *testing.T) {
			for i := 0; i < len(test.env); i += 2 {
				key, value := test.env[i], test.env[i+1]
				old, ok := os.LookupEnv(key)
				if value == "" {
					os.Unsetenv(key)
				} else {
					os.Setenv(key, value)
				}
				if ok {
					defer os.Setenv(key, old)
				} else {
					defer os.Unsetenv(key)
				}
			}

			pkgs, err := packages.Load(nil, "golist")
			if err != nil {
				t.Fatal(err)
			}

			gotIds := make([]string, len(pkgs))
			for i, pkg := range pkgs {
				gotIds[i] = pkg.ID
			}
			if fmt.Sprint(pkgs) != test.wantIDs {
				t.Errorf("got %v; want %v", gotIds, test.wantIDs)
			}
		})
	}
}

func errorMessages(errors []packages.Error) []string {
	var msgs []string
	for _, err := range errors {
		msgs = append(msgs, err.Msg)
	}
	return msgs
}

func srcs(p *packages.Package) []string {
	return cleanPaths(append(p.GoFiles, p.OtherFiles...))
}

// cleanPaths attempts to reduce path names to stable forms
func cleanPaths(paths []string) []string {
	result := make([]string, len(paths))
	for i, src := range paths {
		// The default location for cache data is a subdirectory named go-build
		// in the standard user cache directory for the current operating system.
		if strings.Contains(filepath.ToSlash(src), "/go-build/") {
			result[i] = fmt.Sprintf("%d.go", i) // make cache names predictable
		} else {
			result[i] = filepath.Base(src)
		}
	}
	return result
}

// importGraph returns the import graph as a user-friendly string,
// and a map containing all packages keyed by ID.
func importGraph(initial []*packages.Package) (string, map[string]*packages.Package) {
	out := new(bytes.Buffer)

	initialSet := make(map[*packages.Package]bool)
	for _, p := range initial {
		initialSet[p] = true
	}

	// We can't use Visit because we need to prune
	// the traversal of specific edges, not just nodes.
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
	return p.Types.Scope().Lookup(name).(*types.Const)
}
