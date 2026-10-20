// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"bytes"
	"internal/testenv"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"cmd/compile/internal/importer"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// TestEjectMatchesLowering is the A2 oracle for M1-S6: native and ejected
// programs come from the same lowered nodes. For every .lgo file in
// test/logos/run it parses and type-checks with Config.Logos, calls
// Lower, splices the edits into the original bytes with ApplyEdits, and
// requires syntax.String of the lowered file to equal syntax.String of
// the re-parsed ejected text. A text-level check requires every comment
// of the original source to appear verbatim in the ejected text, because
// AST printing alone cannot prove comment preservation (A7).
func TestEjectMatchesLowering(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	dir := filepath.Join(testenv.GOROOT(t), "test", "logos", "run")
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".lgo") {
			names = append(names, e.Name())
		}
	}
	if len(names) == 0 {
		t.Fatalf("no .lgo files in %s", dir)
	}
	for _, name := range names {
		t.Run(name, func(t *testing.T) {
			ejectCheckFile(t, dir, name)
		})
	}
}

// ejectCheckFile runs the A2 oracle on one run fixture: parse, check,
// lower, splice, compare the printed trees, and verify comment text.
func ejectCheckFile(t *testing.T, dir, name string) {
	t.Helper()
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
	back := ejectParse(t, syntax.NewFileBase(name+".go"), ejected, syntax.CheckBranches)
	if syntax.String(file) != syntax.String(back) {
		t.Errorf("ejected text re-parses to a different tree for %s:\nlowered:\n%s\nejected:\n%s", name, syntax.String(file), syntax.String(back))
	}
	ejectExpectComments(t, name, src, ejected)
}

// ejectParse parses src and fails the test on any syntax error.
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

// ejectCheck type-checks one parsed loGo/s file with Config.Logos and
// fails the test on any type error. It returns the package and info for Lower.
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
			if types2.LogosWarnOnly(err) { // M6-S3: warn-severity rules never fail eject.
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

// ejectLower lowers one checked file and returns its eject edits.
func ejectLower(t *testing.T, pkg *types2.Package, info *types2.Info, file *syntax.File, src []byte) []Edit {
	t.Helper()
	_, edits := Lower(pkg, info, []*syntax.File{file}, map[*syntax.File][]byte{file: src})
	return edits[file]
}

// ejectExpectComments fails the test when any comment of the original
// source is missing verbatim from the ejected text.
func ejectExpectComments(t *testing.T, name string, src, ejected []byte) {
	t.Helper()
	for _, c := range ejectComments(src) {
		if !bytes.Contains(ejected, c) {
			t.Errorf("%s: ejected text drops comment %q", name, c)
		}
	}
}

// ejectComments returns every // and /* */ comment of src in order,
// delimiters included. String, raw-string, and rune literals never
// contribute comments.
func ejectComments(src []byte) [][]byte {
	var out [][]byte
	i := 0
	for i < len(src) {
		switch src[i] {
		case '"', '\'', '`':
			i = ejectQuoted(src, i)
		case '/':
			i = ejectSlash(src, i, &out)
		default:
			i++
		}
	}
	return out
}

// ejectQuoted skips the literal starting at src[i] and returns the index
// after it. Raw literals end at the next backquote; the other forms
// honor backslash escapes.
func ejectQuoted(src []byte, i int) int {
	quote := src[i]
	i++
	for i < len(src) {
		c := src[i]
		if c == '\\' && quote != '`' {
			i += 2
			continue
		}
		if c == quote {
			return i + 1
		}
		i++
	}
	return i
}

// ejectSlash records the comment starting at src[i] when that slash
// opens one, and returns the index after the comment. A lone slash
// advances by one.
func ejectSlash(src []byte, i int, out *[][]byte) int {
	if i+1 >= len(src) {
		return i + 1
	}
	switch src[i+1] {
	case '/':
		end := bytes.IndexByte(src[i:], '\n')
		if end < 0 {
			end = len(src) - i
		}
		*out = append(*out, bytes.TrimSuffix(src[i:i+end], []byte{'\r'}))
		return i + end
	case '*':
		end := bytes.Index(src[i+2:], []byte("*/"))
		if end < 0 {
			*out = append(*out, src[i:])
			return len(src)
		}
		*out = append(*out, src[i:i+2+end+2])
		return i + 2 + end + 2
	default:
		return i + 1
	}
}

// ejectImporter resolves imports through compiled export data, the same
// mechanism the types2 tests use. GOROOT packages, including logos/rt,
// resolve through the installed toolchain.
type ejectImporter struct {
	pkgs map[string]*types2.Package
}

// Import implements types2.Importer.
func (m *ejectImporter) Import(path string) (*types2.Package, error) {
	return m.ImportFrom(path, "", 0)
}

// ImportFrom implements types2.ImporterFrom.
func (m *ejectImporter) ImportFrom(path, srcDir string, mode types2.ImportMode) (*types2.Package, error) {
	return importer.Import(m.pkgs, path, srcDir, nil)
}
