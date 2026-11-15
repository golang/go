// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package facts

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"cmd/internal/goobj"
)

func checkFixture(t *testing.T, src string) (*types2.Package, *types2.Info, *syntax.File) {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "facts.lgo")
	if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	file, err := syntax.ParseFile(path, func(err error) { t.Fatalf("parse: %v", err) }, nil, syntax.LogosMode)
	if err != nil {
		t.Fatal(err)
	}
	var checkErrs []error
	conf := types2.Config{
		Error: func(err error) { checkErrs = append(checkErrs, err) },
		Logos: true,
	}
	info := &types2.Info{
		Types:      make(map[syntax.Expr]types2.TypeAndValue),
		Defs:       make(map[*syntax.Name]types2.Object),
		Uses:       make(map[*syntax.Name]types2.Object),
		Selections: make(map[*syntax.SelectorExpr]*types2.Selection),
	}
	pkg, _ := conf.Check("factstest", []*syntax.File{file}, info)
	for _, err := range checkErrs {
		t.Errorf("check: %v", err)
	}
	if len(checkErrs) > 0 {
		t.FailNow()
	}
	return pkg, info, file
}

func TestWideMutationMaskHelpers(t *testing.T) {
	var m []byte
	for _, bit := range []int{0, 5, 63, 64, 70} {
		m = setMaskBit(m, bit)
	}
	for _, bit := range []int{0, 5, 63, 64, 70} {
		if !maskBit(m, bit) {
			t.Errorf("maskBit(%d) = false, want set", bit)
		}
	}
	for _, bit := range []int{1, 62, 65, 71, 1000} {
		if maskBit(m, bit) {
			t.Errorf("maskBit(%d) = true, want clear", bit)
		}
	}
	if b, ok := parseHexMask(""); ok || b != nil {
		t.Errorf(`parseHexMask("") = %v, %v; want nil, false`, b, ok)
	}
	if b, ok := parseHexMask("0"); !ok || b == nil || len(b) != 0 {
		t.Errorf(`parseHexMask("0") = %v, %v; want non-nil empty, true`, b, ok)
	}
	if b, ok := parseHexMask("ab"); !ok || len(b) != 1 || b[0] != 0xab {
		t.Errorf(`parseHexMask("ab") = %v, %v`, b, ok)
	}
	if got := orHexMask("0", nil); got != "0" {
		t.Errorf(`orHexMask("0", nil) = %q, want "0"`, got)
	}
	if got := orHexMask("0", setMaskBit(nil, 70)); got != "400000000000000000" {
		t.Errorf(`orHexMask("0", bit70) = %q, want 400000000000000000`, got)
	}
}

func TestCalleeFactsMutationTriState(t *testing.T) {
	fp := goobj.FingerprintType{1, 2, 3, 4, 5, 6, 7, 8}
	const path = "example.com/m/trie"
	reg := func(f *Facts) {
		t.Helper()
		member := Encode(fp, f)
		if _, err := Import(path, "export-bytes"+string(fp[:]), member); err != nil {
			t.Fatalf("Import: %v", err)
		}
	}
	pkg := types2.NewPackage("local/pkg", "pkg")
	dep := types2.NewPackage(path, "m")
	fn := types2.NewFunc(syntax.Pos{}, dep, "F", nil)
	s := &constBodyScan{pkg: pkg}

	reg(&Facts{Fns: []FnFact{{Object: "F", Readonly: true}}})
	if _, mask := s.calleeFacts(fn); mask != nil {
		t.Fatalf("readonly-only record: mask = %v, want nil (unknown)", mask)
	}
	reg(&Facts{Fns: []FnFact{{Object: "F", HasAssert: true, AssertTP: "0", AssertRTP: "0"}}})
	if _, mask := s.calleeFacts(fn); mask != nil {
		t.Fatalf("assertion-only record: mask = %v, want nil (unknown)", mask)
	}
	reg(&Facts{Fns: []FnFact{{Object: "F", HasAssert: true, AssertTP: "0", AssertRTP: "0", ParamMutated: "0"}}})
	_, mask := s.calleeFacts(fn)
	if mask == nil || len(mask) != 0 {
		t.Fatalf("ParamMutated=0: mask = %v, want exact empty", mask)
	}
	reg(&Facts{Fns: []FnFact{{Object: "F", ParamMutated: "400000000000000000"}}})
	_, mask = s.calleeFacts(fn)
	if mask == nil || !maskBit(mask, 70) || maskBit(mask, 0) {
		t.Fatalf("wide mask: mask = %v, want bit 70 only", mask)
	}
	reg(&Facts{Fns: []FnFact{{Object: "F", MayYield: true, HasMayYield: true}}})
	if _, mask = s.calleeFacts(fn); mask != nil {
		t.Fatalf("MayYield-only record: mask = %v, want nil (unknown)", mask)
	}
}

func TestConstFnRenderVarStructType(t *testing.T) {
	const src = `package constfntest

var V struct {
	X int
	Y int /* } */
}

const func Total() int {
	return 1
}
`
	pkg, info, file := checkFixture(t, src)
	s := newConstFnSrc(pkg, info, []*syntax.File{file}, map[*syntax.File][]byte{file: []byte(src)})
	s.indexAll()
	if !s.at(file) {
		t.Fatal("cannot load fixture file")
	}
	var vd *syntax.VarDecl
	for _, d := range file.DeclList {
		if v, ok := d.(*syntax.VarDecl); ok {
			vd = v
		}
	}
	if vd == nil {
		t.Fatal("no var declaration in fixture")
	}
	text, err := s.renderVar(file, vd, "V")
	if err != nil {
		t.Fatalf("renderVar: %v", err)
	}
	var errs []string
	_, first := syntax.Parse(syntax.NewFileBase("constfnvar.lgo"), strings.NewReader("package constfntest\n"+text+"\n"), func(err error) {
		errs = append(errs, err.Error())
	}, nil, syntax.CheckBranches|syntax.LogosMode)
	if first != nil || len(errs) > 0 {
		t.Fatalf("rendered var does not parse: %v %q\n%s", first, errs, text)
	}
}
