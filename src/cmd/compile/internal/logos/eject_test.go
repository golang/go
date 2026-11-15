// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"bytes"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"cmd/compile/internal/importer"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func ejectParse(t *testing.T, base *syntax.PosBase, src []byte, mode syntax.Mode) *syntax.File {
	t.Helper()
	var errs []error
	file, _ := syntax.Parse(base, bytes.NewReader(src), func(err error) {
		errs = append(errs, err)
	}, nil, mode)
	for _, err := range errs {
		t.Errorf("parse %s: %v", base.Filename(), err)
	}
	if len(errs) > 0 {
		t.FailNow()
	}
	if file == nil {
		t.Fatalf("parse %s: nil file without errors", base.Filename())
	}
	return file
}

func ejectCheck(t *testing.T, file *syntax.File) (*types2.Package, *types2.Info) {
	t.Helper()
	info := &types2.Info{
		Types:      make(map[syntax.Expr]types2.TypeAndValue),
		Instances:  make(map[*syntax.Name]types2.Instance),
		Defs:       make(map[*syntax.Name]types2.Object),
		Uses:       make(map[*syntax.Name]types2.Object),
		Implicits:  make(map[syntax.Node]types2.Object),
		Selections: make(map[*syntax.SelectorExpr]*types2.Selection),
		Scopes:     make(map[syntax.Node]*types2.Scope),
	}
	var errs []error
	conf := &types2.Config{
		Importer: &ejectImporter{pkgs: make(map[string]*types2.Package)},
		Error: func(err error) {
			if types2.LogosWarnOnly(err) {
				return
			}
			errs = append(errs, err)
		},
		Logos: true,
	}
	pkg, _ := conf.Check("main", []*syntax.File{file}, info)
	for _, err := range errs {
		t.Errorf("check: %v", err)
	}
	if len(errs) > 0 {
		t.FailNow()
	}
	return pkg, info
}

func ejectLower(t *testing.T, pkg *types2.Package, info *types2.Info, file *syntax.File, src []byte) []Edit {
	t.Helper()
	_, edits := Lower(pkg, info, []*syntax.File{file}, map[*syntax.File][]byte{file: src})
	return edits[file]
}

type ejectImporter struct {
	pkgs map[string]*types2.Package
}

func (m *ejectImporter) Import(path string) (*types2.Package, error) {
	return m.ImportFrom(path, "", 0)
}

func (m *ejectImporter) ImportFrom(path, srcDir string, mode types2.ImportMode) (*types2.Package, error) {
	return importer.Import(m.pkgs, path, srcDir, nil)
}
func TestEjectDeferTryJoinBehavior(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	dir := filepath.Join(testenv.GOROOT(t), "test", "logos", "run")
	const name = "defer_try.lgo"
	src, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		t.Fatal(err)
	}
	base := syntax.NewFileBase(filepath.Join(dir, name))
	file := ejectParse(t, base, src, syntax.LogosMode|syntax.CheckBranches)
	pkg, info := ejectCheck(t, file)
	edits := ejectLower(t, pkg, info, file, src)
	ejected, err := ApplyEdits(src, base, edits)
	if err != nil {
		t.Fatalf("ApplyEdits %s: %v", name, err)
	}
	work := t.TempDir()
	if err := os.WriteFile(filepath.Join(work, "go.mod"), []byte("module ejectjoin\n\ngo 1.27\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(work, "main.go"), ejected, 0o644); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(testenv.GoToolPath(t), "run", ".")
	cmd.Dir = work
	cmd.Env = append(os.Environ(), "GOPROXY=off", "GOTOOLCHAIN=local")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("ejected run: %v\n%s", err, out)
	}
	want, err := os.ReadFile(filepath.Join(dir, "defer_try.out"))
	if err != nil {
		t.Fatal(err)
	}
	if got := string(out); got != string(want) {
		t.Fatalf("ejected verdicts differ:\ngot:\n%swant:\n%s", got, want)
	}
}

func TestArenaEjectBlankUseBoundaries(t *testing.T) {
	for _, body := range []string{
		"var (r = open(); r2 = open()); use(r); use(r2)",
		"var r, r2 = open(), open(); use(r); use(r2)",
		"r := open(); r2 := open(); use(r); use(r2)",
		"if r := open(); true { use(r) } else if r2 := open(); true { use(r2) }",
		"for r := open(); false; { use(r) }",
		"switch r := open(); { default: use(r) }",
		"switch r := open(); { case true: use(r); fallthrough; default: }",
		"switch r := open(); r == open() {}",
		"label: switch r := open(); { default: use(r); break label }",
	} {
		t.Run(body, func(t *testing.T) { testArenaBlankUseBoundary(t, body) })
	}
}

func testArenaBlankUseBoundary(t *testing.T, body string) {
	src := []byte("package arena; type Region struct{}; func open() Region { return Region{} }; func use(Region) {}; var (global = open()); func f() { " + body + " }")
	base := syntax.NewFileBase("blank.go")
	file := ejectParse(t, base, src, 0)
	info := &types2.Info{Defs: make(map[*syntax.Name]types2.Object)}
	conf := &types2.Config{}
	pkg, err := conf.Check("logos/arena", []*syntax.File{file}, info)
	if err != nil {
		t.Fatal(err)
	}
	fl := &fileLower{src: src, start: lineStarts(src), file: file, pkg: &packageLower{tpkg: pkg, info: info}}
	a := &arenaEject{fl: fl}
	a.insertBlankUses()
	removeArenaTestUses(fl, file)
	for _, sp := range fl.splices {
		if sp.text != "" && sp.start != sp.end {
			t.Fatal("blank use replaced original source")
		}
		if strings.Contains(sp.text, "\n") {
			t.Fatal("blank use changed original line positions")
		}
	}
	got, err := applySplices(src, fl.splices)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(got), "_ = global") {
		t.Fatal("package variable received a local blank use")
	}
	lowered := ejectParse(t, base, got, 0)
	if _, err := conf.Check("logos/arena", []*syntax.File{lowered}, nil); err != nil {
		t.Fatalf("blank-use output: %v\n%s", err, got)
	}
}

func removeArenaTestUses(fl *fileLower, file *syntax.File) {
	syntax.Inspect(file, func(n syntax.Node) bool {
		s, ok := n.(*syntax.ExprStmt)
		if !ok {
			return true
		}
		call, ok := s.X.(*syntax.CallExpr)
		if ok {
			if name, ok := call.Fun.(*syntax.Name); ok && name.Value == "use" {
				fl.add(fl.off(syntax.StartPos(call)), fl.endOff(call), "")
			}
		}
		return true
	})
}
