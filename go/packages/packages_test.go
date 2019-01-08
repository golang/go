// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packages_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go/ast"
	constantpkg "go/constant"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/packages/packagestest"
)

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

func TestLoadImportsGraph(t *testing.T) { packagestest.TestAll(t, testLoadImportsGraph) }
func testLoadImportsGraph(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a.go":             `package a; const A = 1`,
			"b/b.go":             `package b; import ("golang.org/fake/a"; _ "errors"); var B = a.A`,
			"c/c.go":             `package c; import (_ "golang.org/fake/b"; _ "unsafe")`,
			"c/c2.go":            "// +build ignore\n\n" + `package c; import _ "fmt"`,
			"subdir/d/d.go":      `package d`,
			"subdir/d/d_test.go": `package d; import _ "math/bits"`,
			"subdir/d/x_test.go": `package d_test; import _ "golang.org/fake/subdir/d"`, // TODO(adonovan): test bad import here
			"subdir/e/d.go":      `package e`,
			"e/e.go":             `package main; import _ "golang.org/fake/b"`,
			"e/e2.go":            `package main; import _ "golang.org/fake/c"`,
			"f/f.go":             `package f`,
		}}})
	defer exported.Cleanup()
	exported.Config.Mode = packages.LoadImports
	initial, err := packages.Load(exported.Config, "golang.org/fake/c", "golang.org/fake/subdir/d", "golang.org/fake/e")
	if err != nil {
		t.Fatal(err)
	}

	// Check graph topology.
	graph, all := importGraph(initial)
	wantGraph := `
  errors
  golang.org/fake/a
  golang.org/fake/b
* golang.org/fake/c
* golang.org/fake/e
* golang.org/fake/subdir/d
  unsafe
  golang.org/fake/b -> errors
  golang.org/fake/b -> golang.org/fake/a
  golang.org/fake/c -> golang.org/fake/b
  golang.org/fake/c -> unsafe
  golang.org/fake/e -> golang.org/fake/b
  golang.org/fake/e -> golang.org/fake/c
`[1:]

	if graph != wantGraph {
		t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
	}

	exported.Config.Tests = true
	initial, err = packages.Load(exported.Config, "golang.org/fake/c", "golang.org/fake/subdir/d", "golang.org/fake/e")
	if err != nil {
		t.Fatal(err)
	}

	// Check graph topology.
	graph, all = importGraph(initial)
	wantGraph = `
  errors
  golang.org/fake/a
  golang.org/fake/b
* golang.org/fake/c
* golang.org/fake/e
* golang.org/fake/subdir/d
* golang.org/fake/subdir/d [golang.org/fake/subdir/d.test]
* golang.org/fake/subdir/d.test
* golang.org/fake/subdir/d_test [golang.org/fake/subdir/d.test]
  math/bits
  unsafe
  golang.org/fake/b -> errors
  golang.org/fake/b -> golang.org/fake/a
  golang.org/fake/c -> golang.org/fake/b
  golang.org/fake/c -> unsafe
  golang.org/fake/e -> golang.org/fake/b
  golang.org/fake/e -> golang.org/fake/c
  golang.org/fake/subdir/d [golang.org/fake/subdir/d.test] -> math/bits
  golang.org/fake/subdir/d.test -> golang.org/fake/subdir/d [golang.org/fake/subdir/d.test]
  golang.org/fake/subdir/d.test -> golang.org/fake/subdir/d_test [golang.org/fake/subdir/d.test]
  golang.org/fake/subdir/d.test -> os (pruned)
  golang.org/fake/subdir/d.test -> testing (pruned)
  golang.org/fake/subdir/d.test -> testing/internal/testdeps (pruned)
  golang.org/fake/subdir/d_test [golang.org/fake/subdir/d.test] -> golang.org/fake/subdir/d [golang.org/fake/subdir/d.test]
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
		{"golang.org/fake/a", "a", "package", "a.go"},
		{"golang.org/fake/b", "b", "package", "b.go"},
		{"golang.org/fake/c", "c", "package", "c.go"}, // c2.go is ignored
		{"golang.org/fake/e", "main", "command", "e.go e2.go"},
		{"errors", "errors", "package", "errors.go"},
		{"golang.org/fake/subdir/d", "d", "package", "d.go"},
		{"golang.org/fake/subdir/d.test", "main", "command", "0.go"},
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
	if initial, err := packages.Load(exported.Config, exported.File("golang.org/fake", "c/c.go")); len(initial) == 0 {
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
		initial, err = packages.Load(exported.Config, "golang.org/fake/subdir/...")
		if err != nil {
			t.Fatal(err)
		}
		graph, all = importGraph(initial)
		wantGraph = `
* golang.org/fake/subdir/d
* golang.org/fake/subdir/d [golang.org/fake/subdir/d.test]
* golang.org/fake/subdir/d.test
* golang.org/fake/subdir/d_test [golang.org/fake/subdir/d.test]
* golang.org/fake/subdir/e
  math/bits
  golang.org/fake/subdir/d [golang.org/fake/subdir/d.test] -> math/bits
  golang.org/fake/subdir/d.test -> golang.org/fake/subdir/d [golang.org/fake/subdir/d.test]
  golang.org/fake/subdir/d.test -> golang.org/fake/subdir/d_test [golang.org/fake/subdir/d.test]
  golang.org/fake/subdir/d.test -> os (pruned)
  golang.org/fake/subdir/d.test -> testing (pruned)
  golang.org/fake/subdir/d.test -> testing/internal/testdeps (pruned)
  golang.org/fake/subdir/d_test [golang.org/fake/subdir/d.test] -> golang.org/fake/subdir/d [golang.org/fake/subdir/d.test]
`[1:]

		if graph != wantGraph {
			t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
		}
	}
}

func TestLoadImportsTestVariants(t *testing.T) { packagestest.TestAll(t, testLoadImportsTestVariants) }
func testLoadImportsTestVariants(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a.go":       `package a; import _ "golang.org/fake/b"`,
			"b/b.go":       `package b`,
			"b/b_test.go":  `package b`,
			"b/bx_test.go": `package b_test; import _ "golang.org/fake/a"`,
		}}})
	defer exported.Cleanup()
	exported.Config.Mode = packages.LoadImports
	exported.Config.Tests = true

	initial, err := packages.Load(exported.Config, "golang.org/fake/a", "golang.org/fake/b")
	if err != nil {
		t.Fatal(err)
	}

	// Check graph topology.
	graph, _ := importGraph(initial)
	wantGraph := `
* golang.org/fake/a
  golang.org/fake/a [golang.org/fake/b.test]
* golang.org/fake/b
* golang.org/fake/b [golang.org/fake/b.test]
* golang.org/fake/b.test
* golang.org/fake/b_test [golang.org/fake/b.test]
  golang.org/fake/a -> golang.org/fake/b
  golang.org/fake/a [golang.org/fake/b.test] -> golang.org/fake/b [golang.org/fake/b.test]
  golang.org/fake/b.test -> golang.org/fake/b [golang.org/fake/b.test]
  golang.org/fake/b.test -> golang.org/fake/b_test [golang.org/fake/b.test]
  golang.org/fake/b.test -> os (pruned)
  golang.org/fake/b.test -> testing (pruned)
  golang.org/fake/b.test -> testing/internal/testdeps (pruned)
  golang.org/fake/b_test [golang.org/fake/b.test] -> golang.org/fake/a [golang.org/fake/b.test]
`[1:]

	if graph != wantGraph {
		t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
	}
}

func TestLoadAbsolutePath(t *testing.T) {
	exported := packagestest.Export(t, packagestest.GOPATH, []packagestest.Module{{
		Name: "golang.org/gopatha",
		Files: map[string]interface{}{
			"a/a.go": `package a`,
		}}, {
		Name: "golang.org/gopathb",
		Files: map[string]interface{}{
			"b/b.go": `package b`,
		}}})
	defer exported.Cleanup()

	initial, err := packages.Load(exported.Config, filepath.Dir(exported.File("golang.org/gopatha", "a/a.go")), filepath.Dir(exported.File("golang.org/gopathb", "b/b.go")))
	if err != nil {
		t.Fatalf("failed to load imports: %v", err)
	}

	got := []string{}
	for _, p := range initial {
		got = append(got, p.ID)
	}
	sort.Strings(got)
	want := []string{"golang.org/gopatha/a", "golang.org/gopathb/b"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("initial packages loaded: got [%s], want [%s]", got, want)
	}
}

func TestVendorImports(t *testing.T) {
	exported := packagestest.Export(t, packagestest.GOPATH, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a.go":          `package a; import _ "b"; import _ "golang.org/fake/c";`,
			"a/vendor/b/b.go": `package b; import _ "golang.org/fake/c"`,
			"c/c.go":          `package c; import _ "b"`,
			"c/vendor/b/b.go": `package b`,
		}}})
	defer exported.Cleanup()
	exported.Config.Mode = packages.LoadImports
	initial, err := packages.Load(exported.Config, "golang.org/fake/a", "golang.org/fake/c")
	if err != nil {
		t.Fatal(err)
	}

	graph, all := importGraph(initial)
	wantGraph := `
* golang.org/fake/a
  golang.org/fake/a/vendor/b
* golang.org/fake/c
  golang.org/fake/c/vendor/b
  golang.org/fake/a -> golang.org/fake/a/vendor/b
  golang.org/fake/a -> golang.org/fake/c
  golang.org/fake/a/vendor/b -> golang.org/fake/c
  golang.org/fake/c -> golang.org/fake/c/vendor/b
`[1:]
	if graph != wantGraph {
		t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
	}

	for _, test := range []struct {
		pattern     string
		wantImports string
	}{
		{"golang.org/fake/a", "b:golang.org/fake/a/vendor/b golang.org/fake/c:golang.org/fake/c"},
		{"golang.org/fake/c", "b:golang.org/fake/c/vendor/b"},
		{"golang.org/fake/a/vendor/b", "golang.org/fake/c:golang.org/fake/c"},
		{"golang.org/fake/c/vendor/b", ""},
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

func TestConfigDir(t *testing.T) { packagestest.TestAll(t, testConfigDir) }
func testConfigDir(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a.go":   `package a; const Name = "a" `,
			"a/b/b.go": `package b; const Name = "a/b"`,
			"b/b.go":   `package b; const Name = "b"`,
		}}})
	defer exported.Cleanup()
	aDir := filepath.Dir(exported.File("golang.org/fake", "a/a.go"))
	bDir := filepath.Dir(exported.File("golang.org/fake", "b/b.go"))
	baseDir := filepath.Dir(aDir)

	for _, test := range []struct {
		dir     string
		pattern string
		want    string // value of Name constant
		fails   bool
	}{
		{dir: bDir, pattern: "golang.org/fake/a", want: `"a"`},
		{dir: bDir, pattern: "golang.org/fake/b", want: `"b"`},
		{dir: bDir, pattern: "./a", fails: true},
		{dir: bDir, pattern: "./b", fails: true},
		{dir: baseDir, pattern: "golang.org/fake/a", want: `"a"`},
		{dir: baseDir, pattern: "golang.org/fake/b", want: `"b"`},
		{dir: baseDir, pattern: "./a", want: `"a"`},
		{dir: baseDir, pattern: "./b", want: `"b"`},
		{dir: aDir, pattern: "golang.org/fake/a", want: `"a"`},
		{dir: aDir, pattern: "golang.org/fake/b", want: `"b"`},
		{dir: aDir, pattern: "./a", fails: true},
		{dir: aDir, pattern: "./b", want: `"a/b"`},
	} {
		exported.Config.Mode = packages.LoadSyntax // Use LoadSyntax to ensure that files can be opened.
		exported.Config.Dir = test.dir
		initial, err := packages.Load(exported.Config, test.pattern)
		var got string
		fails := false
		if err != nil {
			fails = true
		} else if len(initial) > 0 {
			if len(initial[0].Errors) > 0 {
				fails = true
			} else if c := constant(initial[0], "Name"); c != nil {
				got = c.Val().String()
			}
		}
		if got != test.want {
			t.Errorf("dir %q, pattern %q: got %s, want %s",
				test.dir, test.pattern, got, test.want)
		}
		if fails != test.fails {
			// TODO: remove when go#28023 is fixed
			if test.fails && strings.HasPrefix(test.pattern, "./") && exporter == packagestest.Modules {
				// Currently go list in module mode does not handle missing directories correctly.
				continue
			}
			t.Errorf("dir %q, pattern %q: error %v, want %v",
				test.dir, test.pattern, fails, test.fails)
		}
	}
}

func TestConfigFlags(t *testing.T) { packagestest.TestAll(t, testConfigFlags) }
func testConfigFlags(t *testing.T, exporter packagestest.Exporter) {
	// Test satisfying +build line tags, with -tags flag.
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			// package a
			"a/a.go": `package a; import _ "golang.org/fake/a/b"`,
			"a/b.go": `// +build tag

package a`,
			"a/c.go": `// +build tag tag2

package a`,
			"a/d.go": `// +build tag,tag2

package a`,
			// package a/b
			"a/b/a.go": `package b`,
			"a/b/b.go": `// +build tag

package b`,
		}}})
	defer exported.Cleanup()

	for _, test := range []struct {
		pattern        string
		tags           []string
		wantSrcs       string
		wantImportSrcs string
	}{
		{`golang.org/fake/a`, []string{}, "a.go", "a.go"},
		{`golang.org/fake/a`, []string{`-tags=tag`}, "a.go b.go c.go", "a.go b.go"},
		{`golang.org/fake/a`, []string{`-tags=tag2`}, "a.go c.go", "a.go"},
		{`golang.org/fake/a`, []string{`-tags=tag tag2`}, "a.go b.go c.go d.go", "a.go b.go"},
	} {
		exported.Config.Mode = packages.LoadImports
		exported.Config.BuildFlags = test.tags

		initial, err := packages.Load(exported.Config, test.pattern)
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

func TestLoadTypes(t *testing.T) { packagestest.TestAll(t, testLoadTypes) }
func testLoadTypes(t *testing.T, exporter packagestest.Exporter) {
	// In LoadTypes and LoadSyntax modes, the compiler will
	// fail to generate an export data file for c, because it has
	// a type error.  The loader should fall back loading a and c
	// from source, but use the export data for b.

	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a.go": `package a; import "golang.org/fake/b"; import "golang.org/fake/c"; const A = "a" + b.B + c.C`,
			"b/b.go": `package b; const B = "b"`,
			"c/c.go": `package c; const C = "c" + 1`,
		}}})
	defer exported.Cleanup()

	exported.Config.Mode = packages.LoadTypes
	initial, err := packages.Load(exported.Config, "golang.org/fake/a")
	if err != nil {
		t.Fatal(err)
	}

	graph, all := importGraph(initial)
	wantGraph := `
* golang.org/fake/a
  golang.org/fake/b
  golang.org/fake/c
  golang.org/fake/a -> golang.org/fake/b
  golang.org/fake/a -> golang.org/fake/c
`[1:]
	if graph != wantGraph {
		t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
	}

	for _, test := range []struct {
		id         string
		wantSyntax bool
	}{
		{"golang.org/fake/a", true},  // need src, no export data for c
		{"golang.org/fake/b", false}, // use export data
		{"golang.org/fake/c", true},  // need src, no export data for c
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

func TestLoadSyntaxOK(t *testing.T) { packagestest.TestAll(t, testLoadSyntaxOK) }
func testLoadSyntaxOK(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a.go": `package a; import "golang.org/fake/b"; const A = "a" + b.B`,
			"b/b.go": `package b; import "golang.org/fake/c"; const B = "b" + c.C`,
			"c/c.go": `package c; import "golang.org/fake/d"; const C = "c" + d.D`,
			"d/d.go": `package d; import "golang.org/fake/e"; const D = "d" + e.E`,
			"e/e.go": `package e; import "golang.org/fake/f"; const E = "e" + f.F`,
			"f/f.go": `package f; const F = "f"`,
		}}})
	defer exported.Cleanup()

	exported.Config.Mode = packages.LoadSyntax
	initial, err := packages.Load(exported.Config, "golang.org/fake/a", "golang.org/fake/c")
	if err != nil {
		t.Fatal(err)
	}

	graph, all := importGraph(initial)
	wantGraph := `
* golang.org/fake/a
  golang.org/fake/b
* golang.org/fake/c
  golang.org/fake/d
  golang.org/fake/e
  golang.org/fake/f
  golang.org/fake/a -> golang.org/fake/b
  golang.org/fake/b -> golang.org/fake/c
  golang.org/fake/c -> golang.org/fake/d
  golang.org/fake/d -> golang.org/fake/e
  golang.org/fake/e -> golang.org/fake/f
`[1:]
	if graph != wantGraph {
		t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
	}

	for _, test := range []struct {
		id           string
		wantSyntax   bool
		wantComplete bool
	}{
		{"golang.org/fake/a", true, true},   // source package
		{"golang.org/fake/b", true, true},   // source package
		{"golang.org/fake/c", true, true},   // source package
		{"golang.org/fake/d", false, true},  // export data package
		{"golang.org/fake/e", false, false}, // export data package
		{"golang.org/fake/f", false, false}, // export data package
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
	aA := constant(all["golang.org/fake/a"], "A")
	if aA == nil {
		t.Fatalf("a.A: got nil")
	}
	if got, want := fmt.Sprintf("%v %v", aA, aA.Val()), `const golang.org/fake/a.A untyped string "abcdef"`; got != want {
		t.Errorf("a.A: got %s, want %s", got, want)
	}
}

func TestLoadDiamondTypes(t *testing.T) { packagestest.TestAll(t, testLoadDiamondTypes) }
func testLoadDiamondTypes(t *testing.T, exporter packagestest.Exporter) {
	// We make a diamond dependency and check the type d.D is the same through both paths
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a.go": `package a; import ("golang.org/fake/b"; "golang.org/fake/c"); var _ = b.B == c.C`,
			"b/b.go": `package b; import "golang.org/fake/d"; var B d.D`,
			"c/c.go": `package c; import "golang.org/fake/d"; var C d.D`,
			"d/d.go": `package d; type D int`,
		}}})
	defer exported.Cleanup()

	exported.Config.Mode = packages.LoadSyntax
	initial, err := packages.Load(exported.Config, "golang.org/fake/a")
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
* golang.org/fake/a
  golang.org/fake/b
  golang.org/fake/c
  golang.org/fake/d
  golang.org/fake/a -> golang.org/fake/b
  golang.org/fake/a -> golang.org/fake/c
  golang.org/fake/b -> golang.org/fake/d
  golang.org/fake/c -> golang.org/fake/d
`[1:]
	if graph != wantGraph {
		t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
	}
}

func TestLoadSyntaxError(t *testing.T) { packagestest.TestAll(t, testLoadSyntaxError) }
func testLoadSyntaxError(t *testing.T, exporter packagestest.Exporter) {
	// A type error in a lower-level package (e) prevents go list
	// from producing export data for all packages that depend on it
	// [a-e]. Only f should be loaded from export data, and the rest
	// should be IllTyped.
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a.go": `package a; import "golang.org/fake/b"; const A = "a" + b.B`,
			"b/b.go": `package b; import "golang.org/fake/c"; const B = "b" + c.C`,
			"c/c.go": `package c; import "golang.org/fake/d"; const C = "c" + d.D`,
			"d/d.go": `package d; import "golang.org/fake/e"; const D = "d" + e.E`,
			"e/e.go": `package e; import "golang.org/fake/f"; const E = "e" + f.F + 1`, // type error
			"f/f.go": `package f; const F = "f"`,
		}}})
	defer exported.Cleanup()

	exported.Config.Mode = packages.LoadSyntax
	initial, err := packages.Load(exported.Config, "golang.org/fake/a", "golang.org/fake/c")
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
		{"golang.org/fake/a", true, true},
		{"golang.org/fake/b", true, true},
		{"golang.org/fake/c", true, true},
		{"golang.org/fake/d", true, true},
		{"golang.org/fake/e", true, true},
		{"golang.org/fake/f", false, false},
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
	aA := constant(all["golang.org/fake/a"], "A")
	if aA == nil {
		t.Fatalf("a.A: got nil")
	}
	if got, want := aA.String(), `const golang.org/fake/a.A invalid type`; got != want {
		t.Errorf("a.A: got %s, want %s", got, want)
	}
}

// This function tests use of the ParseFile hook to modify
// the AST after parsing.
func TestParseFileModifyAST(t *testing.T) { packagestest.TestAll(t, testParseFileModifyAST) }
func testParseFileModifyAST(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a.go": `package a; const A = "a" `,
		}}})
	defer exported.Cleanup()

	exported.Config.Mode = packages.LoadAllSyntax
	exported.Config.ParseFile = func(fset *token.FileSet, filename string, src []byte) (*ast.File, error) {
		const mode = parser.AllErrors | parser.ParseComments
		f, err := parser.ParseFile(fset, filename, src, mode)
		// modify AST to change `const A = "a"` to `const A = "b"`
		spec := f.Decls[0].(*ast.GenDecl).Specs[0].(*ast.ValueSpec)
		spec.Values[0].(*ast.BasicLit).Value = `"b"`
		return f, err
	}
	initial, err := packages.Load(exported.Config, "golang.org/fake/a")
	if err != nil {
		t.Error(err)
	}

	// Check value of a.A has been set to "b"
	a := initial[0]
	got := constant(a, "A").Val().String()
	if got != `"b"` {
		t.Errorf("a.A: got %s, want %s", got, `"b"`)
	}
}

func TestOverlay(t *testing.T) { packagestest.TestAll(t, testOverlay) }
func testOverlay(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a.go": `package a; import "golang.org/fake/b"; const A = "a" + b.B`,
			"b/b.go": `package b; import "golang.org/fake/c"; const B = "b" + c.C`,
			"c/c.go": `package c; const C = "c"`,
			"d/d.go": `package d; const D = "d"`,
		}}})
	defer exported.Cleanup()

	for i, test := range []struct {
		overlay  map[string][]byte
		want     string // expected value of a.A
		wantErrs []string
	}{
		{nil, `"abc"`, nil},                 // default
		{map[string][]byte{}, `"abc"`, nil}, // empty overlay
		{map[string][]byte{exported.File("golang.org/fake", "c/c.go"): []byte(`package c; const C = "C"`)}, `"abC"`, nil},
		{map[string][]byte{exported.File("golang.org/fake", "b/b.go"): []byte(`package b; import "golang.org/fake/c"; const B = "B" + c.C`)}, `"aBc"`, nil},
		// Overlay with an existing file in an existing package adding a new import.
		{map[string][]byte{exported.File("golang.org/fake", "b/b.go"): []byte(`package b; import "golang.org/fake/d"; const B = "B" + d.D`)}, `"aBd"`, nil},
		// Overlay with a new file in an existing package.
		{map[string][]byte{
			exported.File("golang.org/fake", "c/c.go"):                                               []byte(`package c;`),
			filepath.Join(filepath.Dir(exported.File("golang.org/fake", "c/c.go")), "c_new_file.go"): []byte(`package c; const C = "Ç"`)},
			`"abÇ"`, nil},
		// Overlay with a new file in an existing package, adding a new dependency to that package.
		{map[string][]byte{
			exported.File("golang.org/fake", "c/c.go"):                                               []byte(`package c;`),
			filepath.Join(filepath.Dir(exported.File("golang.org/fake", "c/c.go")), "c_new_file.go"): []byte(`package c; import "golang.org/fake/d"; const C = "c" + d.D`)},
			`"abcd"`, nil},
	} {
		exported.Config.Overlay = test.overlay
		exported.Config.Mode = packages.LoadAllSyntax
		initial, err := packages.Load(exported.Config, "golang.org/fake/a")
		if err != nil {
			t.Error(err)
			continue
		}

		// Check value of a.A.
		a := initial[0]
		aA := constant(a, "A")
		if aA == nil {
			t.Errorf("%d. a.A: got nil", i)
			continue
		}
		got := aA.Val().String()
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
	packagestest.TestAll(t, testLoadAllSyntaxImportErrors)
}
func testLoadAllSyntaxImportErrors(t *testing.T, exporter packagestest.Exporter) {
	// TODO(matloob): Remove this once go list -e -compiled is fixed.
	// See https://golang.org/issue/26755
	t.Skip("go list -compiled -e fails with non-zero exit status for empty packages")

	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"unicycle/unicycle.go": `package unicycle; import _ "unicycle"`,
			"bicycle1/bicycle1.go": `package bicycle1; import _ "bicycle2"`,
			"bicycle2/bicycle2.go": `package bicycle2; import _ "bicycle1"`,
			"bad/bad.go":           `not a package declaration`,
			"empty/README.txt":     `not a go file`,
			"root/root.go": `package root
import (
	_ "bicycle1"
	_ "unicycle"
	_ "nonesuch"
	_ "empty"
	_ "bad"
)`,
		}}})
	defer exported.Cleanup()

	exported.Config.Mode = packages.LoadAllSyntax
	initial, err := packages.Load(exported.Config, "root")
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

func TestAbsoluteFilenames(t *testing.T) { packagestest.TestAll(t, testAbsoluteFilenames) }
func testAbsoluteFilenames(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a.go":          `package a; const A = 1`,
			"b/b.go":          `package b; import ("golang.org/fake/a"; _ "errors"); var B = a.A`,
			"b/vendor/a/a.go": `package a; const A = 1`,
			"c/c.go":          `package c; import (_ "golang.org/fake/b"; _ "unsafe")`,
			"c/c2.go":         "// +build ignore\n\n" + `package c; import _ "fmt"`,
			"subdir/d/d.go":   `package d`,
			"subdir/e/d.go":   `package e`,
			"e/e.go":          `package main; import _ "golang.org/fake/b"`,
			"e/e2.go":         `package main; import _ "golang.org/fake/c"`,
			"f/f.go":          `package f`,
			"f/f.s":           ``,
		}}})
	defer exported.Cleanup()
	exported.Config.Dir = filepath.Dir(filepath.Dir(exported.File("golang.org/fake", "a/a.go")))

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
		{"golang.org/fake/a", "a.go"},
		{"golang.org/fake/b/vendor/a", "a.go"},
		{"golang.org/fake/b", "b.go"},
		{"golang.org/fake/c", "c.go"},
		{"golang.org/fake/subdir/d", "d.go"},
		{"golang.org/fake/subdir/e", "d.go"},
		{"golang.org/fake/e", "e.go e2.go"},
		{"golang.org/fake/f", "f.go f.s"},
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
		exported.Config.Mode = packages.LoadFiles
		pkgs, err := packages.Load(exported.Config, test.pattern)
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

func TestContains(t *testing.T) { packagestest.TestAll(t, testContains) }
func testContains(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a.go": `package a; import "golang.org/fake/b"`,
			"b/b.go": `package b; import "golang.org/fake/c"`,
			"c/c.go": `package c`,
		}}})
	defer exported.Cleanup()
	bFile := exported.File("golang.org/fake", "b/b.go")
	exported.Config.Mode = packages.LoadImports
	initial, err := packages.Load(exported.Config, "file="+bFile)
	if err != nil {
		t.Fatal(err)
	}

	graph, _ := importGraph(initial)
	wantGraph := `
* golang.org/fake/b
  golang.org/fake/c
  golang.org/fake/b -> golang.org/fake/c
`[1:]
	if graph != wantGraph {
		t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
	}
}

func TestContainsOverlay(t *testing.T) { packagestest.TestAll(t, testContainsOverlay) }
func testContainsOverlay(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a.go": `package a; import "golang.org/fake/b"`,
			"b/b.go": `package b; import "golang.org/fake/c"`,
			"c/c.go": `package c`,
		}}})
	defer exported.Cleanup()
	bOverlayFile := filepath.Join(filepath.Dir(exported.File("golang.org/fake", "b/b.go")), "b_overlay.go")
	exported.Config.Mode = packages.LoadImports
	exported.Config.Overlay = map[string][]byte{bOverlayFile: []byte(`package b;`)}
	initial, err := packages.Load(exported.Config, "file="+bOverlayFile)
	if err != nil {
		t.Fatal(err)
	}

	graph, _ := importGraph(initial)
	wantGraph := `
* golang.org/fake/b
  golang.org/fake/c
  golang.org/fake/b -> golang.org/fake/c
`[1:]
	if graph != wantGraph {
		t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
	}
}

// This test ensures that the effective GOARCH variable in the
// application determines the Sizes function used by the type checker.
// This behavior is a stop-gap until we make the build system's query
// too report the correct sizes function for the actual configuration.
func TestSizes(t *testing.T) { packagestest.TestAll(t, testSizes) }
func testSizes(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a.go": `package a; import "unsafe"; const WordSize = 8*unsafe.Sizeof(int(0))`,
		}}})
	defer exported.Cleanup()

	exported.Config.Mode = packages.LoadSyntax
	savedEnv := exported.Config.Env
	for arch, wantWordSize := range map[string]int64{"386": 32, "amd64": 64} {
		exported.Config.Env = append(savedEnv, "GOARCH="+arch)
		initial, err := packages.Load(exported.Config, "golang.org/fake/a")
		if err != nil {
			t.Fatal(err)
		}
		if packages.PrintErrors(initial) > 0 {
			t.Fatal("there were errors")
		}
		gotWordSize, _ := constantpkg.Int64Val(constant(initial[0], "WordSize").Val())
		if gotWordSize != wantWordSize {
			t.Errorf("for GOARCH=%s, got word size %d, want %d", arch, gotWordSize, wantWordSize)
		}
	}
}

// TestContains_FallbackSticks ensures that when there are both contains and non-contains queries
// the decision whether to fallback to the pre-1.11 go list sticks across both sets of calls to
// go list.
func TestContains_FallbackSticks(t *testing.T) { packagestest.TestAll(t, testContains_FallbackSticks) }
func testContains_FallbackSticks(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a.go": `package a; import "golang.org/fake/b"`,
			"b/b.go": `package b; import "golang.org/fake/c"`,
			"c/c.go": `package c`,
		}}})
	defer exported.Cleanup()

	exported.Config.Mode = packages.LoadImports
	bFile := exported.File("golang.org/fake", "b/b.go")
	initial, err := packages.Load(exported.Config, "golang.org/fake/a", "file="+bFile)
	if err != nil {
		t.Fatal(err)
	}

	graph, _ := importGraph(initial)
	wantGraph := `
* golang.org/fake/a
* golang.org/fake/b
  golang.org/fake/c
  golang.org/fake/a -> golang.org/fake/b
  golang.org/fake/b -> golang.org/fake/c
`[1:]
	if graph != wantGraph {
		t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
	}
}

func TestName(t *testing.T) { packagestest.TestAll(t, testName) }
func testName(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/needle/needle.go":       `package needle; import "golang.org/fake/c"`,
			"b/needle/needle.go":       `package needle;`,
			"c/c.go":                   `package c;`,
			"irrelevant/irrelevant.go": `package irrelevant;`,
		}}})
	defer exported.Cleanup()

	exported.Config.Mode = packages.LoadImports
	initial, err := packages.Load(exported.Config, "name=needle")
	if err != nil {
		t.Fatal(err)
	}
	graph, _ := importGraph(initial)
	wantGraph := `
* golang.org/fake/a/needle
* golang.org/fake/b/needle
  golang.org/fake/c
  golang.org/fake/a/needle -> golang.org/fake/c
`[1:]
	if graph != wantGraph {
		t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
	}
}

func TestName_Modules(t *testing.T) {
	if usesOldGolist {
		t.Skip("pre-modules version of Go")
	}

	// Test the top-level package case described in runNamedQueries.
	// Note that overriding GOPATH below prevents Export from
	// creating more than one module.
	exported := packagestest.Export(t, packagestest.Modules, []packagestest.Module{{
		Name: "golang.org/pkg",
		Files: map[string]interface{}{
			"pkg.go": `package pkg`,
		}}})
	defer exported.Cleanup()

	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	// testdata/TestNamed_Modules contains:
	// - pkg/mod/github.com/heschik/tools-testrepo@v1.0.0/pkg
	// - pkg/mod/github.com/heschik/tools-testrepo/v2@v2.0.0/pkg
	// - src/b/pkg
	exported.Config.Mode = packages.LoadImports
	exported.Config.Env = append(exported.Config.Env, "GOPATH="+wd+"/testdata/TestName_Modules")
	initial, err := packages.Load(exported.Config, "name=pkg")
	if err != nil {
		t.Fatal(err)
	}
	graph, _ := importGraph(initial)
	wantGraph := `
* github.com/heschik/tools-testrepo/pkg
* github.com/heschik/tools-testrepo/v2/pkg
* golang.org/pkg
`[1:]
	if graph != wantGraph {
		t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
	}
}

func TestName_ModulesDedup(t *testing.T) {
	if usesOldGolist {
		t.Skip("pre-modules version of Go")
	}

	exported := packagestest.Export(t, packagestest.Modules, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"fake.go": `package fake`,
		}}})
	defer exported.Cleanup()

	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	// testdata/TestNamed_ModulesDedup contains:
	// - pkg/mod/github.com/heschik/tools-testrepo/v2@v2.0.2/pkg/pkg.go
	// - pkg/mod/github.com/heschik/tools-testrepo/v2@v2.0.1/pkg/pkg.go
	// - pkg/mod/github.com/heschik/tools-testrepo@v1.0.0/pkg/pkg.go
	// but, inexplicably, not v2.0.0. Nobody knows why.
	exported.Config.Mode = packages.LoadImports
	exported.Config.Env = append(exported.Config.Env, "GOPATH="+wd+"/testdata/TestName_ModulesDedup")
	initial, err := packages.Load(exported.Config, "name=pkg")
	if err != nil {
		t.Fatal(err)
	}
	for _, pkg := range initial {
		if strings.Contains(pkg.PkgPath, "v2") {
			if strings.Contains(pkg.GoFiles[0], "v2.0.2") {
				return
			}
		}
	}
	t.Errorf("didn't find v2.0.2 of pkg in Load results: %v", initial)
}

// Test that Load doesn't get confused when two different patterns match the same package. See #29297.
func TestRedundantQueries(t *testing.T) { packagestest.TestAll(t, testRedundantQueries) }
func testRedundantQueries(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a.go": `package a;`,
		}}})
	defer exported.Cleanup()

	initial, err := packages.Load(exported.Config, "errors", "name=errors")
	if err != nil {
		t.Fatal(err)
	}
	if len(initial) != 1 || initial[0].Name != "errors" {
		t.Fatalf(`Load("errors", "name=errors") = %v, wanted just the errors package`, initial)
	}
}

// Test that Load with no patterns is equivalent to loading "." via the golist
// driver.
func TestNoPatterns(t *testing.T) { packagestest.TestAll(t, testNoPatterns) }
func testNoPatterns(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a.go":   `package a;`,
			"a/b/b.go": `package b;`,
		}}})
	defer exported.Cleanup()

	aDir := filepath.Dir(exported.File("golang.org/fake", "a/a.go"))
	exported.Config.Dir = aDir

	initial, err := packages.Load(exported.Config)
	if err != nil {
		t.Fatal(err)
	}
	if len(initial) != 1 || initial[0].Name != "a" {
		t.Fatalf(`Load() = %v, wanted just the package in the current directory`, initial)
	}
}

func TestJSON(t *testing.T) { packagestest.TestAll(t, testJSON) }
func testJSON(t *testing.T, exporter packagestest.Exporter) {
	//TODO: add in some errors
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a.go": `package a; const A = 1`,
			"b/b.go": `package b; import "golang.org/fake/a"; var B = a.A`,
			"c/c.go": `package c; import "golang.org/fake/b" ; var C = b.B`,
			"d/d.go": `package d; import "golang.org/fake/b" ; var D = b.B`,
		}}})
	defer exported.Cleanup()

	exported.Config.Mode = packages.LoadImports
	initial, err := packages.Load(exported.Config, "golang.org/fake/c", "golang.org/fake/d")
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
	"ID": "golang.org/fake/a",
	"Name": "a",
	"PkgPath": "golang.org/fake/a",
	"GoFiles": [
		"a.go"
	],
	"CompiledGoFiles": [
		"a.go"
	]
}
{
	"ID": "golang.org/fake/b",
	"Name": "b",
	"PkgPath": "golang.org/fake/b",
	"GoFiles": [
		"b.go"
	],
	"CompiledGoFiles": [
		"b.go"
	],
	"Imports": {
		"golang.org/fake/a": "golang.org/fake/a"
	}
}
{
	"ID": "golang.org/fake/c",
	"Name": "c",
	"PkgPath": "golang.org/fake/c",
	"GoFiles": [
		"c.go"
	],
	"CompiledGoFiles": [
		"c.go"
	],
	"Imports": {
		"golang.org/fake/b": "golang.org/fake/b"
	}
}
{
	"ID": "golang.org/fake/d",
	"Name": "d",
	"PkgPath": "golang.org/fake/d",
	"GoFiles": [
		"d.go"
	],
	"CompiledGoFiles": [
		"d.go"
	],
	"Imports": {
		"golang.org/fake/b": "golang.org/fake/b"
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
		ID:   "golang.org/fake/a",
		Name: "a",
	}, {
		ID:   "golang.org/fake/b",
		Name: "b",
		Imports: map[string]*packages.Package{
			"golang.org/fake/a": {ID: "golang.org/fake/a"},
		},
	}, {
		ID:   "golang.org/fake/c",
		Name: "c",
		Imports: map[string]*packages.Package{
			"golang.org/fake/b": {ID: "golang.org/fake/b"},
		},
	}, {
		ID:   "golang.org/fake/d",
		Name: "d",
		Imports: map[string]*packages.Package{
			"golang.org/fake/b": {ID: "golang.org/fake/b"},
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

func TestRejectInvalidQueries(t *testing.T) {
	queries := []string{"key=", "key=value"}
	cfg := &packages.Config{
		Mode: packages.LoadImports,
		Env:  append(os.Environ(), "GO111MODULE=off"),
	}
	for _, q := range queries {
		if _, err := packages.Load(cfg, q); err == nil {
			t.Errorf("packages.Load(%q) succeeded. Expected \"invalid query type\" error", q)
		} else if !strings.Contains(err.Error(), "invalid query type") {
			t.Errorf("packages.Load(%q): got error %v, want \"invalid query type\" error", q, err)
		}
	}
}

func TestPatternPassthrough(t *testing.T) { packagestest.TestAll(t, testPatternPassthrough) }
func testPatternPassthrough(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a.go": `package a;`,
		}}})
	defer exported.Cleanup()

	initial, err := packages.Load(exported.Config, "pattern=a")
	if err != nil {
		t.Fatal(err)
	}

	graph, _ := importGraph(initial)
	wantGraph := `
* a
`[1:]
	if graph != wantGraph {
		t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
	}

}

func TestConfigDefaultEnv(t *testing.T) { packagestest.TestAll(t, testConfigDefaultEnv) }
func testConfigDefaultEnv(t *testing.T, exporter packagestest.Exporter) {
	const driverJSON = `{
  "Roots": ["gopackagesdriver"],
  "Packages": [{"ID": "gopackagesdriver", "Name": "gopackagesdriver"}]
}`
	var (
		pathKey      string
		driverScript packagestest.Writer
	)
	switch runtime.GOOS {
	case "windows":
		// TODO(jayconrod): write an equivalent batch script for windows.
		// Hint: "type" can be used to read a file to stdout.
		t.Skip("test requires sh")
	case "plan9":
		pathKey = "path"
		driverScript = packagestest.Script(`#!/bin/rc

cat <<'EOF'
` + driverJSON + `
EOF
`)
	default:
		pathKey = "PATH"
		driverScript = packagestest.Script(`#!/bin/sh

cat - <<'EOF'
` + driverJSON + `
EOF
`)
	}
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"bin/gopackagesdriver": driverScript,
			"golist/golist.go":     "package golist",
		}}})
	defer exported.Cleanup()
	driver := exported.File("golang.org/fake", "bin/gopackagesdriver")
	binDir := filepath.Dir(driver)
	if err := os.Chmod(driver, 0755); err != nil {
		t.Fatal(err)
	}

	path, ok := os.LookupEnv(pathKey)
	var pathWithDriver string
	if ok {
		pathWithDriver = binDir + string(os.PathListSeparator) + path
	} else {
		pathWithDriver = binDir
	}
	coreEnv := exported.Config.Env
	for _, test := range []struct {
		desc    string
		path    string
		driver  string
		wantIDs string
	}{
		{
			desc:    "driver_off",
			path:    pathWithDriver,
			driver:  "off",
			wantIDs: "[golist]",
		}, {
			desc:    "driver_unset",
			path:    pathWithDriver,
			driver:  "",
			wantIDs: "[gopackagesdriver]",
		}, {
			desc:    "driver_set",
			path:    "",
			driver:  driver,
			wantIDs: "[gopackagesdriver]",
		},
	} {
		t.Run(test.desc, func(t *testing.T) {
			oldPath := os.Getenv(pathKey)
			os.Setenv(pathKey, test.path)
			defer os.Setenv(pathKey, oldPath)
			exported.Config.Env = append(coreEnv, "GOPACKAGESDRIVER="+test.driver)
			pkgs, err := packages.Load(exported.Config, "golist")
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

// This test that a simple x test package layout loads correctly.
// There was a bug in go list where it returned multiple copies of the same
// package (specifically in this case of golang.org/fake/a), and this triggered
// a bug in go/packages where it would leave an empty entry in the root package
// list. This would then cause a nil pointer crash.
// This bug was triggered by the simple package layout below, and thus this
// test will make sure the bug remains fixed.
func TestBasicXTest(t *testing.T) { packagestest.TestAll(t, testBasicXTest) }
func testBasicXTest(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a.go":      `package a;`,
			"a/a_test.go": `package a_test;`,
		}}})
	defer exported.Cleanup()

	exported.Config.Mode = packages.LoadFiles
	exported.Config.Tests = true
	_, err := packages.Load(exported.Config, "golang.org/fake/a")
	if err != nil {
		t.Fatal(err)
	}
}

func TestErrorMissingFile(t *testing.T) { packagestest.TestAll(t, testErrorMissingFile) }
func testErrorMissingFile(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a_test.go": `package a;`,
		}}})
	defer exported.Cleanup()

	exported.Config.Mode = packages.LoadSyntax
	exported.Config.Tests = false
	pkgs, err := packages.Load(exported.Config, "missing.go")
	if err != nil {
		t.Fatal(err)
	}
	t.Log(pkgs)
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
		// If the source file doesn't have an extension like .go or .s,
		// it comes from GOCACHE. The names there aren't predictable.
		name := filepath.Base(src)
		if !strings.Contains(name, ".") {
			result[i] = fmt.Sprintf("%d.go", i) // make cache names predictable
		} else {
			result[i] = name
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
				// math/bits took on a dependency on unsafe in 1.12, which breaks some
				// tests. As a short term hack, prune that edge.
				// TODO(matloob): think of a cleaner solution, or remove math/bits from the test.
				if p.ID == "math/bits" && imp.ID == "unsafe" {
					continue
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

func constant(p *packages.Package, name string) *types.Const {
	if p == nil || p.Types == nil {
		return nil
	}
	c := p.Types.Scope().Lookup(name)
	if c == nil {
		return nil
	}
	return c.(*types.Const)
}
