// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"errors"
	"go/build"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	gcimporter "cmd/compile/internal/importer"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// runLogosDir locates GOROOT/test/logos/run from this test's own file,
// independent of the working directory the test runs under.
func runLogosDir(t *testing.T) string {
	t.Helper()
	_, self, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	root := filepath.Dir(self)
	for range 5 {
		root = filepath.Dir(root)
	}
	dir := filepath.Join(root, "test", "logos", "run")
	if st, err := os.Stat(dir); err != nil || !st.IsDir() {
		t.Fatalf("logos run dir not found: %s", dir)
	}
	return dir
}

// testImporter resolves imports for TestNoLogosNodesSurvive. Packages with
// export data in the build tree come through gcimporter, the pattern used
// by cmd/compile/internal/types2 tests; anything without export data yet,
// such as logos/rt, is checked from GOROOT source instead.
type testImporter struct {
	pkgs map[string]*types2.Package
	busy map[string]bool
}

func (m *testImporter) Import(path string) (*types2.Package, error) {
	return m.ImportFrom(path, "", 0)
}

func (m *testImporter) ImportFrom(path, srcDir string, mode types2.ImportMode) (*types2.Package, error) {
	if path == "unsafe" {
		return types2.Unsafe, nil
	}
	if pkg := m.pkgs[path]; pkg != nil && pkg.Complete() {
		return pkg, nil
	}
	pkg, gcerr := gcimporter.Import(m.pkgs, path, srcDir, nil)
	if gcerr == nil {
		return pkg, nil
	}
	srcpkg, srcerr := m.importFromSource(path)
	if srcerr != nil {
		return nil, errors.Join(gcerr, srcerr)
	}
	return srcpkg, nil
}

func (m *testImporter) importFromSource(path string) (*types2.Package, error) {
	if m.busy[path] {
		return nil, errors.New("import cycle: " + path)
	}
	dir := filepath.Join(runtime.GOROOT(), "src", filepath.FromSlash(path))
	p, err := build.ImportDir(dir, 0)
	if err != nil {
		return nil, err
	}
	if len(p.GoFiles) == 0 {
		return nil, errors.New("no Go files: " + path)
	}
	m.busy[path] = true
	defer delete(m.busy, path)
	files, err := parseGoFiles(p, dir)
	if err != nil {
		return nil, err
	}
	conf, errs := checkConfig(m)
	info := &types2.Info{
		Defs: make(map[*syntax.Name]types2.Object),
		Uses: make(map[*syntax.Name]types2.Object),
	}
	pkg, _ := conf.Check(path, files, info)
	if err := errors.Join(*errs...); err != nil {
		return nil, err
	}
	m.pkgs[path] = pkg
	return pkg, nil
}

func parseGoFiles(p *build.Package, dir string) ([]*syntax.File, error) {
	var files []*syntax.File
	var errs []error
	for _, name := range p.GoFiles {
		f, err := syntax.ParseFile(filepath.Join(dir, name), func(err error) { errs = append(errs, err) }, nil, 0)
		if err != nil {
			return nil, errors.Join(append(errs, err)...)
		}
		files = append(files, f)
	}
	return files, nil
}

func checkConfig(m *testImporter) (types2.Config, *[]error) {
	errs := new([]error)
	conf := types2.Config{
		Importer: m,
		Error:    func(err error) { *errs = append(*errs, err) },
	}
	return conf, errs
}

// countLogosNodes counts surviving loGo/s nodes under root.
func countLogosNodes(root syntax.Node) int {
	n := 0
	syntax.Inspect(root, func(x syntax.Node) bool {
		switch x.(type) {
		case *syntax.TryExpr, *syntax.ErrdeferStmt:
			n++
		}
		return true
	})
	return n
}

// TestNoLogosNodesSurvive lowers every test/logos/run file and asserts the
// lowered files hold no TryExpr or ErrdeferStmt (invariant A1).
func TestNoLogosNodesSurvive(t *testing.T) {
	dir := runLogosDir(t)
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	imp := &testImporter{pkgs: make(map[string]*types2.Package), busy: make(map[string]bool)}
	found := 0
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".lgo") {
			continue
		}
		found++
		t.Run(e.Name(), func(t *testing.T) {
			checkLoweredClean(t, imp, filepath.Join(dir, e.Name()))
		})
	}
	if found == 0 {
		t.Fatalf("no .lgo files in %s", dir)
	}
}

// checkLoweredClean parses, checks, and lowers one file, then asserts no
// TryExpr or ErrdeferStmt survives in the lowered files and that every
// edit applies to the original source.
func checkLoweredClean(t *testing.T, imp *testImporter, path string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var perrs []error
	f, err := syntax.ParseFile(path, func(err error) { perrs = append(perrs, err) }, nil, syntax.CheckBranches|syntax.LogosMode)
	if err != nil {
		t.Fatal(errors.Join(append(perrs, err)...))
	}
	before := countLogosNodes(f)
	info := &types2.Info{
		Defs:       make(map[*syntax.Name]types2.Object),
		Uses:       make(map[*syntax.Name]types2.Object),
		Types:      make(map[syntax.Expr]types2.TypeAndValue),
		Selections: make(map[*syntax.SelectorExpr]*types2.Selection),
		Scopes:     make(map[syntax.Node]*types2.Scope),
	}
	var cerrs []error
	conf := types2.Config{
		Importer: imp,
		Error:    func(err error) { cerrs = append(cerrs, err) },
		Logos:    true,
	}
	files := []*syntax.File{f}
	src := map[*syntax.File][]byte{f: data}
	pkg, _ := conf.Check("main", files, info)
	if err := errors.Join(cerrs...); err != nil {
		t.Fatalf("type check %s: %v", path, err)
	}
	changed, edits := Lower(pkg, info, files, src)
	if after := countLogosNodes(f); after != 0 {
		t.Fatalf("Lower left %d loGo/s nodes in %s", after, path)
	}
	if before != 0 && !changed {
		t.Fatalf("Lower reported no change for %s holding loGo/s nodes", path)
	}
	out, err := ApplyEdits(data, f.Pos().FileBase(), edits[f])
	if err != nil {
		t.Fatalf("ApplyEdits %s: %v", path, err)
	}
	if after := countLogosNodes(reparseLowered(t, path, out)); after != 0 {
		t.Fatalf("Lower left %d loGo/s nodes in applied edits of %s", after, path)
	}
}

// reparseLowered parses applied edit output from a scratch .lgo file.
func reparseLowered(t *testing.T, path string, out []byte) *syntax.File {
	t.Helper()
	tmp := filepath.Join(t.TempDir(), filepath.Base(path))
	if err := os.WriteFile(tmp, out, 0666); err != nil {
		t.Fatal(err)
	}
	var perrs []error
	g, err := syntax.ParseFile(tmp, func(err error) { perrs = append(perrs, err) }, nil, syntax.CheckBranches|syntax.LogosMode)
	if err != nil {
		t.Fatal(errors.Join(append(perrs, err)...))
	}
	return g
}
