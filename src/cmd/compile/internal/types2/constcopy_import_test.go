// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

// Consumer behavior proof that copied const evaluation carries the
// check's import capability: a copied call to an imported recursive
// const func reaches the selected comptime::ctfe_step_limit, a
// below-limit copied call succeeds, and evaluation without a threaded
// importer still fails closed (no global cache). Plain Go checking is
// unaffected.

import (
	"fmt"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
)

const logosCopyImportPath = "example.com/copyimportrec"

const logosCopyImportRecSrc = "package reclib\n\nconst func Rec(n int) int {\n\tif n <= 0 {\n\t\treturn 0\n\t}\n\treturn 1 + Rec(n-1)\n}\n"

// logosCopyImportStub is the configured importer a copy checker opens
// packages through. It serves only packages the test installed.
type logosCopyImportStub struct {
	pkgs map[string]*Package
}

func (s *logosCopyImportStub) Import(path string) (*Package, error) {
	if p, ok := s.pkgs[path]; ok {
		return p, nil
	}
	return nil, fmt.Errorf("unknown import %s", path)
}

// logosCopyImportFixture registers the constfn record for the reclib
// Rec helper and serves its defining package through a stub importer.
func logosCopyImportFixture(t *testing.T) (*Package, Importer) {
	t.Helper()
	defPkg := NewPackage(logosCopyImportPath, "reclib")
	sig := NewSignatureType(nil, nil, nil,
		NewTuple(NewVar(nopos, defPkg, "n", Typ[Int])),
		NewTuple(NewVar(nopos, defPkg, "", Typ[Int])), false)
	defPkg.Scope().Insert(NewFunc(nopos, defPkg, "Rec", sig))
	LogosImportedConstFuncSource = func(path, object string) []byte {
		if path == logosCopyImportPath && object == "Rec" {
			return []byte(logosCopyImportRecSrc)
		}
		return nil
	}
	return defPkg, &logosCopyImportStub{pkgs: map[string]*Package{logosCopyImportPath: defPkg}}
}

// logosCopyCaller checks a trivial caller package under the selected
// step budget and returns its package, file, and info, mirroring the
// main check that records importer and context for copies.
func logosCopyCaller(t *testing.T, imp Importer) (*Package, *syntax.File, *Info) {
	t.Helper()
	file, err := syntax.Parse(syntax.NewFileBase("caller"), strings.NewReader("package caller\nfunc Use() int { return 1 }\n"), nil, nil, 0)
	if err != nil {
		t.Fatalf("parse caller: %v", err)
	}
	conf := &Config{
		Logos:             true,
		Importer:          imp,
		Context:           NewContext(),
		LogosCTFEMaxSteps: 100,
		Sizes:             SizesFor("gc", "amd64"),
	}
	info := &Info{
		Types: map[syntax.Expr]TypeAndValue{}, Defs: map[*syntax.Name]Object{},
		Uses: map[*syntax.Name]Object{}, Selections: map[*syntax.SelectorExpr]*Selection{},
		Scopes: map[syntax.Node]*Scope{}, Instances: map[*syntax.Name]Instance{},
		// The driver records the selected budget on both conf and
		// info; the copy keeps the selected limit through ctx.
		LogosCTFEMaxSteps: 100,
	}
	pkg, err := conf.Check("example.com/copycaller", []*syntax.File{file}, info)
	if err != nil {
		t.Fatalf("caller check: %v", err)
	}
	return pkg, file, info
}

// logosCopyParseFunc parses one copy-shaped function declaration.
func logosCopyParseFunc(t *testing.T, src string) *syntax.FuncDecl {
	t.Helper()
	file, err := syntax.Parse(syntax.NewFileBase("cfcopy"), strings.NewReader(src), nil, nil, 0)
	if err != nil {
		t.Fatalf("parse copy: %v", err)
	}
	for _, decl := range file.DeclList {
		if fn, ok := decl.(*syntax.FuncDecl); ok {
			return fn
		}
	}
	t.Fatal("copy source declares no function")
	return nil
}

func logosCopyErrText(errs []Error) string {
	var sb strings.Builder
	for _, err := range errs {
		sb.WriteString(err.Msg)
		sb.WriteString("\n")
	}
	return sb.String()
}

// TestLogosCopyImportedStepLimit proves a copied call to the imported
// recursive const func reaches the selected step limit instead of
// failing to open the defining package.
func TestLogosCopyImportedStepLimit(t *testing.T) {
	defPkg, imp := logosCopyImportFixture(t)
	pkg, file, info := logosCopyCaller(t, imp)
	fn := logosCopyParseFunc(t, "package cfcopy\nfunc CopyOver() int {\nconst Limit = _logos_t0.Rec(1000)\nreturn Limit\n}\n")
	ctx := LogosCopyContext{
		Scope:    info.Scopes[file],
		Aliases:  map[*Package]string{defPkg: "_logos_t0"},
		MaxSteps: info.LogosCTFEMaxSteps,
		Importer: info.LogosImporter,
		Context:  info.LogosContext,
	}
	if ctx.Scope == nil {
		t.Fatal("caller file scope missing")
	}
	_, errs := LogosCheckCopyFunc(pkg, info, ctx, fn)
	got := logosCopyErrText(errs)
	if !strings.Contains(got, "comptime::ctfe_step_limit") {
		t.Fatalf("copied Rec(1000) errors = %q, want the selected step limit", got)
	}
	if strings.Contains(got, "no importer installed") {
		t.Fatalf("copied Rec(1000) errors = %q, want import resolution, not a missing importer", got)
	}
}

// TestLogosCopyImportedUnderLimit proves a below-limit copied call to
// the same imported const func succeeds and records its value.
func TestLogosCopyImportedUnderLimit(t *testing.T) {
	defPkg, imp := logosCopyImportFixture(t)
	pkg, file, info := logosCopyCaller(t, imp)
	fn := logosCopyParseFunc(t, "package cfcopy\nfunc CopyUnder() int {\nconst Limit = _logos_t0.Rec(5)\nreturn Limit\n}\n")
	ctx := LogosCopyContext{
		Scope:    info.Scopes[file],
		Aliases:  map[*Package]string{defPkg: "_logos_t0"},
		MaxSteps: info.LogosCTFEMaxSteps,
		Importer: info.LogosImporter,
		Context:  info.LogosContext,
	}
	obj, errs := LogosCheckCopyFunc(pkg, info, ctx, fn)
	if len(errs) != 0 {
		t.Fatalf("copied Rec(5) errors = %q, want success", logosCopyErrText(errs))
	}
	if obj == nil {
		t.Fatal("copied Rec(5) returned no function object")
	}
	var limit *Const
	for _, o := range info.Defs {
		if c, ok := o.(*Const); ok && c.Name() == "Limit" {
			limit = c
		}
	}
	if limit == nil {
		t.Fatal("copied Rec(5) recorded no Limit constant")
	}
	v, ok := info.LogosCTFE[limit]
	if !ok || !v.IsValid() {
		t.Fatal("copied Rec(5) recorded no value for Limit")
	}
	if v.Int64() != 5 {
		t.Fatalf("copied Rec(5) Limit = %d, want 5", v.Int64())
	}
}

// TestLogosCopyWithoutImporterFailsClosed proves the capability is
// threaded, not ambient: the same over-limit copy without an importer
// in context or info fails to open the package instead of evaluating.
func TestLogosCopyWithoutImporterFailsClosed(t *testing.T) {
	defPkg, _ := logosCopyImportFixture(t)
	pkg := NewPackage("example.com/copycallerplain", "caller")
	info := &Info{
		Types: map[syntax.Expr]TypeAndValue{}, Defs: map[*syntax.Name]Object{},
		Uses: map[*syntax.Name]Object{}, Selections: map[*syntax.SelectorExpr]*Selection{},
		Scopes: map[syntax.Node]*Scope{}, Instances: map[*syntax.Name]Instance{},
	}
	fn := logosCopyParseFunc(t, "package cfcopy\nfunc CopyOver() int {\nconst Limit = _logos_t0.Rec(1000)\nreturn Limit\n}\n")
	ctx := LogosCopyContext{
		Scope:    pkg.Scope(),
		Aliases:  map[*Package]string{defPkg: "_logos_t0"},
		MaxSteps: 100,
	}
	_, errs := LogosCheckCopyFunc(pkg, info, ctx, fn)
	got := logosCopyErrText(errs)
	if !strings.Contains(got, "no importer installed") {
		t.Fatalf("importer-less copy errors = %q, want the missing-importer failure", got)
	}
	if strings.Contains(got, "ctfe_step_limit") {
		t.Fatalf("importer-less copy errors = %q, want no step-limit evaluation", got)
	}
}

// TestLogosPlainCheckUnaffected proves ordinary Go checking still
// passes with no Logos importer or context recorded.
func TestLogosPlainCheckUnaffected(t *testing.T) {
	file, err := syntax.Parse(syntax.NewFileBase("plain"), strings.NewReader("package p\nconst C = 40 + 2\nfunc F() int { return C }\n"), nil, nil, 0)
	if err != nil {
		t.Fatalf("parse plain: %v", err)
	}
	info := &Info{
		Types: map[syntax.Expr]TypeAndValue{}, Defs: map[*syntax.Name]Object{},
		Uses: map[*syntax.Name]Object{},
	}
	if _, err := (&Config{}).Check("example.com/plain", []*syntax.File{file}, info); err != nil {
		t.Fatalf("plain Go check: %v", err)
	}
}
