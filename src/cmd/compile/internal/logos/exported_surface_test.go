// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"encoding/json"
	"strings"
	"testing"

	"cmd/compile/internal/logos/ctfe"
	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func TestSurfaceUnionMessageNamesUnion(t *testing.T) {
	got := DualUseMessage("U")
	if !strings.Contains(got, "U") {
		t.Errorf("DualUseMessage = %q, want the union name", got)
	}
	if !strings.Contains(got, "constraint") || !strings.Contains(got, "Go cannot express both") {
		t.Errorf("DualUseMessage = %q, want the constraint conflict", got)
	}
}

func TestSurfaceUnknownSymbolCarriesIdentity(t *testing.T) {
	err := &UnknownSymbolError{Pkg: "p", Symbol: "s"}
	if !strings.Contains(err.Error(), "p") || !strings.Contains(err.Error(), "s") {
		t.Errorf("UnknownSymbolError = %q, want the package and symbol", err)
	}
}

func TestSurfaceRodataRegistryRoundTrip(t *testing.T) {
	const pkg, name = "example.com/weakgo-surface", "V"
	RegisterRodata(pkg, name)
	if !IsRodata(pkg, name) {
		t.Error("IsRodata missed a registered variable")
	}
	if IsRodata(pkg, "W") {
		t.Error("IsRodata matched an unregistered name")
	}
	if IsRodata("example.com/weakgo-other", name) {
		t.Error("IsRodata matched an unregistered package")
	}
}

func TestSurfaceUnionPredicates(t *testing.T) {
	union := types2.NewUnion([]*types2.Term{types2.NewTerm(true, types2.Typ[types2.Int])})
	if !IsUnionTerm(union) {
		t.Error("IsUnionTerm missed a union with terms")
	}
	if IsUnionTerm(types2.Typ[types2.Int]) {
		t.Error("IsUnionTerm matched a basic type")
	}
	if HasUnionTerms(types2.NewInterfaceType(nil, nil)) {
		t.Error("HasUnionTerms matched an empty interface")
	}
	if !HasUnionTerms(types2.NewInterfaceType(nil, []types2.Type{union})) {
		t.Error("HasUnionTerms missed an embedded union")
	}
}

func TestSurfaceClassifyKeepsValueUnions(t *testing.T) {
	const src = "package p\n\ntype Ints interface {\n\t~int | ~string\n}\n\nfunc Sum[T Ints](xs []T) T {\n\tvar z T\n\treturn z\n}\n"
	file, err := syntax.Parse(syntax.NewFileBase("unions.lgo"), strings.NewReader(src), nil, nil, syntax.LogosMode)
	if err != nil {
		t.Fatal(err)
	}
	info := &types2.Info{
		Defs: make(map[*syntax.Name]types2.Object),
		Uses: make(map[*syntax.Name]types2.Object),
	}
	conf := types2.Config{Importer: newTestImporter(t), Logos: true}
	if _, err := conf.Check("p", []*syntax.File{file}, info); err != nil {
		t.Fatal(err)
	}
	modes := ClassifyDeclaredUnions(info, []*syntax.File{file})
	if len(modes) != 1 {
		t.Fatalf("ClassifyDeclaredUnions found %d unions, want one", len(modes))
	}
	for obj, use := range modes {
		if use != UnionConstraintOnly {
			t.Errorf("use of %s = %v, want constraint only", obj.Name(), use)
		}
		if KeepUnionDecl(modes, obj) {
			t.Errorf("KeepUnionDecl kept constraint-only %s", obj.Name())
		}
	}
}

func TestSurfaceLowerGoCallsNilGuard(t *testing.T) {
	if LowerGoCalls(nil, nil, nil) {
		t.Error("LowerGoCalls with no package must report nothing to lower")
	}
}

func TestSurfaceConstMatInitMiss(t *testing.T) {
	file, err := syntax.Parse(syntax.NewFileBase("x.lgo"), strings.NewReader("package p\nvar x = 1\n"), nil, nil, syntax.LogosMode)
	if err != nil {
		t.Fatal(err)
	}
	var lit syntax.Expr
	syntax.Inspect(file, func(n syntax.Node) bool {
		if b, ok := n.(*syntax.BasicLit); ok && lit == nil {
			lit = b
		}
		return true
	})
	if lit == nil {
		t.Fatal("no literal found")
	}

	if _, ok := LookupConstMatInit("p", lit); ok {
		t.Error("LookupConstMatInit matched an unregistered expression")
	}
}

var _ UnionUse = UnionConstraintOnly

func TestSurfaceApplyReadonlyNilGuard(t *testing.T) {
	var noFacts []facts.ReadonlyFact
	if err := ApplyReadonlyFactsInPlace(nil, noFacts); err != nil {
		t.Errorf("ApplyReadonlyFactsInPlace(nil, nil) = %v, want nil", err)
	}
}

func TestSurfaceBuildMatLiteralNil(t *testing.T) {
	if got := BuildMatLiteral(ctfe.Value{}, types2.Typ[types2.Int], syntax.Pos{}); got != nil {
		t.Errorf("BuildMatLiteral of a nil value = %v, want nil", got)
	}
}

func unionSurfaceCheck(t *testing.T, src string) (*types2.Info, []*syntax.File, map[*syntax.File][]byte) {
	t.Helper()
	file, err := syntax.Parse(syntax.NewFileBase("unions.lgo"), strings.NewReader(src), nil, nil, syntax.LogosMode)
	if err != nil {
		t.Fatal(err)
	}
	info := &types2.Info{
		Defs: make(map[*syntax.Name]types2.Object),
		Uses: make(map[*syntax.Name]types2.Object),
	}
	conf := types2.Config{Importer: newTestImporter(t), Logos: true}
	if _, err := conf.Check("p", []*syntax.File{file}, info); err != nil {
		t.Fatal(err)
	}
	files := []*syntax.File{file}
	return info, files, map[*syntax.File][]byte{file: []byte(src)}
}

func TestSurfaceEjectUnionDualError(t *testing.T) {
	const src = "package p\n\ntype U interface {\n\tint | string\n}\n\nfunc F[T U](x T) T {\n\treturn x\n}\n\nvar _ U\n"
	info, files, srcs := unionSurfaceCheck(t, src)
	if _, err := EjectUnionEdits(info, files, srcs); err == nil || !strings.Contains(err.Error(), "U") {
		t.Errorf("EjectUnionEdits dual use err = %v, want the union name", err)
	}
}

func TestSurfaceEjectUnionConstraintOnly(t *testing.T) {
	const src = "package p\n\ntype Ints interface {\n\t~int | ~string\n}\n\nfunc Sum[T Ints](xs []T) T {\n\tvar z T\n\treturn z\n}\n"
	info, files, srcs := unionSurfaceCheck(t, src)
	if _, err := EjectUnionEdits(info, files, srcs); err != nil {
		t.Errorf("EjectUnionEdits constraint-only err = %v, want nil", err)
	}
}

func TestSurfaceNativeUnionValueEdits(t *testing.T) {
	const src = "package p\n\ntype U interface {\n\tint | string\n}\n\nvar _ U\n"
	info, files, srcs := unionSurfaceCheck(t, src)
	edits := NativeUnionEdits(info, files, srcs)
	if _, ok := edits[files[0]]; !ok {
		t.Error("NativeUnionEdits recorded no edits for a value-only union")
	}
}

func TestSurfaceHintWireRoundTrip(t *testing.T) {
	hint := UnionHint{Position: HintPos{Line: 3, Character: 7}, Type: "union", Members: []string{"int", "string"}}
	data, err := json.Marshal(hint)
	if err != nil {
		t.Fatal(err)
	}
	var back UnionHint
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatal(err)
	}
	if back.Position != hint.Position || back.Type != hint.Type || len(back.Members) != 2 || back.Members[0] != "int" || back.Members[1] != "string" {
		t.Errorf("UnionHint round trip = %+v, want %+v", back, hint)
	}
	rng := HintRange{Start: HintPos{Line: 1, Character: 2}, End: HintPos{Line: 1, Character: 9}}
	data, err = json.Marshal(rng)
	if err != nil {
		t.Fatal(err)
	}
	var backRange HintRange
	if err := json.Unmarshal(data, &backRange); err != nil {
		t.Fatal(err)
	}
	if backRange != rng {
		t.Errorf("HintRange round trip = %+v, want %+v", backRange, rng)
	}
}

func TestSurfaceNextReportIdentifiesFile(t *testing.T) {
	const src = "package p\n\nfunc errNew() error { return nil }\n\nfunc g() error {\n\terr := errNew()\n\tif err != nil {\n\t\treturn err\n\t}\n\treturn nil\n}\n"
	var perrs []error
	f, err := syntax.Parse(syntax.NewFileBase("next.go"), strings.NewReader(src), func(err error) { perrs = append(perrs, err) }, nil, 0)
	if err != nil {
		t.Fatal(perrs, err)
	}
	info := &types2.Info{
		StoreTypesInSyntax: true,
		Defs:               make(map[*syntax.Name]types2.Object),
		Uses:               make(map[*syntax.Name]types2.Object),
		Selections:         make(map[*syntax.SelectorExpr]*types2.Selection),
		Scopes:             make(map[syntax.Node]*types2.Scope),
	}
	conf := types2.Config{Importer: newTestImporter(t)}
	if _, err := conf.Check("p", []*syntax.File{f}, info); err != nil {
		t.Fatal(err)
	}
	data := []byte(src)
	reps := ScanNext(info, []*syntax.File{f}, map[*syntax.File][]byte{f: data}, map[*syntax.File]string{f: "next.go"})
	if len(reps) != 1 {
		t.Fatalf("ScanNext reports = %+v, want one", reps)
	}
	if reps[0].File != "next.go" || len(reps[0].Sum) != 64 {
		t.Errorf("NextFileReport = %+v, want the file with a sha256 sum", reps[0])
	}
	pkg := NextPackageReport{Version: "1", Package: "p", Files: reps}
	if pkg.Package != "p" || len(pkg.Files) != 1 || pkg.Files[0].File != "next.go" {
		t.Errorf("NextPackageReport = %+v, want the package report", pkg)
	}
}

func TestSurfaceOwnedRegistry(t *testing.T) {
	before := len(OwnedTypeNodes())
	RegisterOwnedNode(nil, types2.LogosOwnedType{})
	if len(OwnedTypeNodes()) != before {
		t.Error("RegisterOwnedNode(nil) changed the registry")
	}
	file, err := syntax.Parse(syntax.NewFileBase("x.lgo"), strings.NewReader("package p\nvar x = 1\n"), nil, nil, syntax.LogosMode)
	if err != nil {
		t.Fatal(err)
	}
	var name syntax.Expr
	syntax.Inspect(file, func(n syntax.Node) bool {
		if id, ok := n.(*syntax.Name); ok && name == nil && id.Value == "x" {
			name = id
		}
		return true
	})
	if name == nil {
		t.Fatal("no name found")
	}
	want := types2.LogosOwnedFactory("example.com/weakgo-surface", "New")
	RegisterOwnedNode(name, want)
	got, ok := OwnedTypeNodes()[name]
	if !ok || got.FactoryPath != want.FactoryPath || got.FactoryName != want.FactoryName {
		t.Errorf("OwnedTypeNodes round trip = %+v, want %+v", got, want)
	}
	first, err := OwnedConstantRequests()
	if err != nil {
		t.Fatal(err)
	}
	second, err := OwnedConstantRequests()
	if err != nil {
		t.Fatal(err)
	}
	if len(first) != len(second) {
		t.Fatalf("OwnedConstantRequests changed size: %d vs %d", len(first), len(second))
	}
	for i := range first {
		if first[i].Owner != second[i].Owner || first[i].Package != second[i].Package || first[i].ID != second[i].ID || len(first[i].Declarations) != len(second[i].Declarations) {
			t.Fatalf("OwnedConstantRequests is not deterministic at %d", i)
		}
	}
}
