// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"strings"
	"testing"
)

func checkEnumPackage(t *testing.T, src string) (*types.Package, error) {
	t.Helper()
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "enum.go", src, parser.SkipObjectResolution)
	if err != nil {
		t.Fatal(err)
	}
	return new(types.Config).Check("p", fset, []*ast.File{file}, nil)
}

func TestEnumTypes(t *testing.T) {
	const src = `package p

type Result enum {
	Ok { value int }
	Err { err error }
	None
}

func (r Result) Value() int {
	switch r {
	case Ok: return r.value
	case Err: return -1
	case None, nil: return 0
	}
	return 0
}
func makeResult() Result { return Result.Ok{value: 42} }
`
	pkg, err := checkEnumPackage(t, src)
	if err != nil {
		t.Fatal(err)
	}
	result := pkg.Scope().Lookup("Result").Type().(*types.Named)
	variants := result.EnumVariants()
	if len(variants) != 3 || variants[0].Obj().Name() != "Result.Ok" || variants[1].Obj().Name() != "Result.Err" || variants[2].Obj().Name() != "Result.None" {
		t.Fatalf("Result variants = %v, want [Result.Ok Result.Err Result.None]", variants)
	}
	ok, errVariant, none := variants[0], variants[1], variants[2]

	if _, ok := result.Underlying().(*types.Interface); !ok {
		t.Fatalf("Result underlying type is %T, want *types.Interface", result.Underlying())
	}
	if !types.Implements(result, result.Underlying().(*types.Interface)) {
		t.Error("Result does not implement its own enum interface")
	}
	for _, variant := range []*types.Named{ok, errVariant, none} {
		if _, ok := variant.Underlying().(*types.Struct); !ok {
			t.Errorf("%s underlying type is %T, want *types.Struct", variant.Obj().Name(), variant.Underlying())
		}
		if !types.AssignableTo(variant, result) {
			t.Errorf("%s is not assignable to Result", variant.Obj().Name())
		}
	}
	if fields := ok.Underlying().(*types.Struct); fields.NumFields() != 1 || fields.Field(0).Name() != "value" {
		t.Fatalf("Ok fields = %s, want value int", fields)
	}
	if method, _, _ := types.LookupFieldOrMethod(result, true, pkg, "Value"); method == nil {
		t.Error("method declared on enum Result was not collected")
	}
	if method, _, _ := types.LookupFieldOrMethod(ok, true, pkg, "Value"); method != nil {
		t.Error("enum method Value unexpectedly belongs to variant Ok")
	}
	if ok.EnumType() != result {
		t.Fatalf("Ok enum type = %v, want Result", ok.EnumType())
	}
	if types.AssignableTo(types.NewPointer(ok), result) {
		t.Error("*Ok is assignable to Result")
	}
	if types.Implements(types.NewPointer(ok), result.Underlying().(*types.Interface)) {
		t.Error("*Ok implements Result")
	}
	forged := types.NewNamed(types.NewTypeName(token.NoPos, pkg, "Forged", nil), types.NewStruct([]*types.Var{
		types.NewField(token.NoPos, pkg, "Ok", ok, true),
	}, nil), nil)
	if types.AssignableTo(forged, result) || types.Implements(forged, result.Underlying().(*types.Interface)) {
		t.Error("type embedding Ok implements Result")
	}
	orSig := types.NewSignatureType(nil, nil, nil,
		types.NewTuple(types.NewVar(token.NoPos, nil, "", types.Typ[types.Int])),
		types.NewTuple(types.NewVar(token.NoPos, nil, "", types.Typ[types.Int])), false)
	orIface := types.NewInterfaceType([]*types.Func{types.NewFunc(token.NoPos, nil, "Value", orSig)}, nil).Complete()
	if types.Implements(result, orIface) {
		t.Error("enum convenience methods must not make Result implement another interface")
	}
}

func TestEnumVariantMethods(t *testing.T) {
	pkg, err := checkEnumPackage(t, `package p
type Decision enum { Allow; Deny { Reason string } }
var _ string = Decision.Allow{}.Variant()
func decisionVariant(decision Decision) string { return decision.Variant() }
`)
	if err != nil {
		t.Fatal(err)
	}
	decision := pkg.Scope().Lookup("Decision").Type().(*types.Named)
	for _, typ := range append([]*types.Named{decision}, decision.EnumVariants()...) {
		method, _, _ := types.LookupFieldOrMethod(typ, true, pkg, "Variant")
		fn, ok := method.(*types.Func)
		if !ok {
			t.Fatalf("%s Variant method = %T, want *types.Func", typ, method)
		}
		sig := fn.Type().(*types.Signature)
		if sig.Params().Len() != 0 || sig.Results().Len() != 1 || sig.Results().At(0).Type() != types.Typ[types.String] {
			t.Fatalf("%s Variant signature = %s, want func() string", typ, sig)
		}
	}
}

func TestEnumVariantMethodCannotBeOverridden(t *testing.T) {
	_, err := checkEnumPackage(t, `package p
type Decision enum { Allow }
func (Decision) Variant() string { return "custom" }
`)
	if err == nil || !strings.Contains(err.Error(), "conflicts with generated enum method Variant") {
		t.Fatalf("Variant override error = %v", err)
	}
}

func TestVariantMethodOnRecursiveNonEnum(t *testing.T) {
	_, err := checkEnumPackage(t, `package p
type A struct { B *B }
type B A
func (B) Variant() string { return "B" }
`)
	if err != nil {
		t.Fatal(err)
	}
}

func TestEnumVersion(t *testing.T) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "enum.go", "package p; type E enum { A }", parser.SkipObjectResolution)
	if err != nil {
		t.Fatal(err)
	}
	_, err = (&types.Config{GoVersion: "go1.27"}).Check("p", fset, []*ast.File{file}, nil)
	if err == nil || !strings.Contains(err.Error(), "requires go1.28 or later") {
		t.Fatalf("enum version error = %v", err)
	}
}

func TestEnumVariantTypeRejected(t *testing.T) {
	const src = `package p
type Result enum { Ok { value int }; Err }
func (o Result.Ok) Value() int { return o.value }
`
	_, err := checkEnumPackage(t, src)
	if err == nil || !strings.Contains(err.Error(), "is a constructor, not a type") {
		t.Fatalf("variant type error = %v", err)
	}
}

func TestEnumMethodVariantCollisionRejected(t *testing.T) {
	const src = `package p
type Result enum { Ok }
func (Result) Ok() {}
`
	_, err := checkEnumPackage(t, src)
	if err == nil || !strings.Contains(err.Error(), "conflicts with enum variant Ok") {
		t.Fatalf("method/variant collision error = %v", err)
	}
}

func TestEnumTypeSwitch(t *testing.T) {
	const exhaustive = `package p
type Result enum { Ok { value int }; Err { err error }; None }
func inspect(r Result) int {
	switch v := r.(type) {
	case Ok: return v.value
	case Err: return len(v.err.Error())
	case None: return 0
	case nil: return -1
	}
	return -1
}
`
	if _, err := checkEnumPackage(t, exhaustive); err != nil {
		t.Fatal(err)
	}

	const missing = `package p
type Result enum { Ok; Err; None }
func inspect(r Result) { switch r.(type) { case Ok: } }
`
	_, err := checkEnumPackage(t, missing)
	if err == nil || !strings.Contains(err.Error(), "non-exhaustive enum switch on Result; missing Err, None, nil") {
		t.Fatalf("non-exhaustive switch error = %v", err)
	}

	const withDefault = `package p
type Result enum { Ok; Err }
func inspect(r Result) { switch r.(type) { case Ok:; default: } }
`
	if _, err := checkEnumPackage(t, withDefault); err != nil {
		t.Fatal(err)
	}
}

func TestEnumValueSwitchNarrowing(t *testing.T) {
	const exhaustive = `package p
type Result enum { Ok { value int }; Err { err error }; None }
func inspect(result Result) int {
	switch result {
	case Ok: return result.value
	case Err: return len(result.err.Error())
	case None: return 0
	case nil: return -1
	}
	return -2
}
`
	if _, err := checkEnumPackage(t, exhaustive); err != nil {
		t.Fatal(err)
	}

	const expressions = `package p
type Result enum { Ok; Err }
func makeResult() Result { return Result.Ok{} }
type Holder struct { Result Result }
func inspect(h Holder) {
	switch makeResult() { case Ok:; case Err:; case nil: }
	switch h.Result { case Ok:; case Err:; case nil: }
}
`
	if _, err := checkEnumPackage(t, expressions); err != nil {
		t.Fatal(err)
	}

	const duplicate = `package p
type Result enum { Ok; Err }
func inspect(result Result) { switch result { case Ok:; case Ok:; case Err:; case nil: } }
`
	_, err := checkEnumPackage(t, duplicate)
	if err == nil || !strings.Contains(err.Error(), "duplicate case Result.Ok in enum switch") {
		t.Fatalf("duplicate enum case error = %v", err)
	}
}

func TestExhaustiveEnumSwitchTerminates(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{
			name: "value switch",
			body: `switch result {
	case Ok: return result.value
	case Err, None, nil: return 0
}`,
		},
		{
			name: "type switch",
			body: `switch result := result.(type) {
	case Ok: return result.value
	case Err, None, nil: return 0
}`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			src := `package p
type Result enum { Ok { value int }; Err; None }
func value(result Result) int {
` + test.body + `
}
`
			if _, err := checkEnumPackage(t, src); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestEnumRejectsPointerVariant(t *testing.T) {
	const src = `package p
type Result enum { Ok }
type R = Result
var _ R = &Result.Ok{}
`
	_, err := checkEnumPackage(t, src)
	if err == nil || !strings.Contains(err.Error(), "pointer to enum variant is not an enum value") {
		t.Fatalf("pointer variant error = %v", err)
	}
}

func TestEnumRejectsPromotedMarker(t *testing.T) {
	tests := []string{`package p
type Inner enum { Public }
type Outer enum { Wrap { Inner } }
		`, `package p
type Inner enum { Public }
type Carrier struct { Inner }
type Outer enum { Wrap { Carrier } }
		`}
	for _, src := range tests {
		_, err := checkEnumPackage(t, src)
		if err == nil || !strings.Contains(err.Error(), "promotes an enum marker") {
			t.Fatalf("promoted enum marker error = %v", err)
		}
	}
}

func TestEnumVariantOnlyAllowedAsConstructor(t *testing.T) {
	valid := `package p
type Result enum { Ok { Value int } }
var _ = Result.Ok{Value: 1}
var _ any = &Result.Ok{Value: 2}
type Option[T any] enum { Some { Value T } }
var _ = Option.Some[int]{Value: 3}
`
	if _, err := checkEnumPackage(t, valid); err != nil {
		t.Fatalf("variant constructors: %v", err)
	}
	for _, use := range []string{
		"type Alias = Result.Ok",
		"var _ Result.Ok",
		"type Embedded struct { Result.Ok }",
		"type Outer enum { Wrap { Result.Ok } }",
		"func f(Result.Ok) {}",
		"var _ = new(Result.Ok)",
		"var _ = Result.Ok(0)",
		"func f(x Result) { switch x.(type) { case Result.Ok: } }",
	} {
		src := "package p\ntype Result enum { Ok }\n" + use
		_, err := checkEnumPackage(t, src)
		if err == nil || !strings.Contains(err.Error(), "is a constructor, not a type") {
			t.Fatalf("%s: variant type error = %v", use, err)
		}
	}
}

func TestGenericEnumTypes(t *testing.T) {
	const src = `package p

type Option[T any] enum {
	Some { value T }
	None
}

func (o Option[T]) Or(zero T) T {
	switch o {
	case Some: return o.value
	case None, nil: return zero
	}
	return zero
}

var _ Option[int] = Option.Some[int]{value: 1}
`
	pkg, err := checkEnumPackage(t, src)
	if err != nil {
		t.Fatal(err)
	}
	option := pkg.Scope().Lookup("Option").Type().(*types.Named)
	variants := option.EnumVariants()
	some, none := variants[0], variants[1]
	optionParam := option.TypeParams().At(0)
	someParam := some.TypeParams().At(0)
	noneParam := none.TypeParams().At(0)
	if optionParam == someParam || optionParam == noneParam || someParam == noneParam {
		t.Fatal("generic enum and variants share type parameter objects")
	}
	if fieldType := some.Underlying().(*types.Struct).Field(0).Type(); fieldType != someParam {
		t.Fatalf("Some.value type = %v, want Some type parameter %v", fieldType, someParam)
	}
	if method, _, _ := types.LookupFieldOrMethod(option, true, pkg, "Or"); method == nil {
		t.Fatal("generic enum method Or was not collected")
	}
	marker := some.Method(0).Type().(*types.Signature)
	recv := marker.Recv().Type().(*types.Named)
	if recv.TypeArgs().Len() != 1 || marker.RecvTypeParams().Len() != 1 {
		t.Fatalf("marker receiver = %v (type args %d, receiver type params %d)", recv, recv.TypeArgs().Len(), marker.RecvTypeParams().Len())
	}
}

func TestEnumVariantsRequireQualification(t *testing.T) {
	const src = `package p
type Result enum { Ok }
var inferred Result = Ok{}
func f() Result { return Ok{} }
var _ = Result.Ok{}
`
	if _, err := checkEnumPackage(t, src); err != nil {
		t.Fatalf("contextual variants: %v", err)
	}

	const invalid = `package p
type Result enum { Ok }
var _ = Ok{}
`
	_, err := checkEnumPackage(t, invalid)
	if err == nil || !strings.Contains(err.Error(), "undefined: Ok") {
		t.Fatalf("bare variant error = %v", err)
	}
}

func TestDuplicateEnumVariant(t *testing.T) {
	const src = `package p
type E enum { A; A }
`
	_, err := checkEnumPackage(t, src)
	if err == nil || !strings.Contains(err.Error(), "A redeclared") {
		t.Fatalf("duplicate variant error = %v, want redeclaration error", err)
	}
}

func TestEnumVariantVisibility(t *testing.T) {
	const src = `package p
type E enum { Public; private }
`
	pkg, err := checkEnumPackage(t, src)
	if err != nil {
		t.Fatal(err)
	}
	variants := pkg.Scope().Lookup("E").Type().(*types.Named).EnumVariants()
	if !variants[0].Obj().Exported() || variants[1].Obj().Exported() {
		t.Fatalf("variant visibility: Public=%v private=%v", variants[0].Obj().Exported(), variants[1].Obj().Exported())
	}
	if got := variants[1].Obj().Id(); got != "p.E.private" {
		t.Fatalf("private variant ID = %q, want p.E.private", got)
	}
}

func TestGenericEnumInstantiationSeal(t *testing.T) {
	const src = `package p
type Option[T any] enum { Some { Value T }; None }
var _ Option[int] = Option.Some[string]{Value: "wrong"}
`
	if _, err := checkEnumPackage(t, src); err == nil || !strings.Contains(err.Error(), "cannot use") {
		t.Fatalf("cross-instantiation assignment error = %v", err)
	}
}

func TestLocalEnumSeal(t *testing.T) {
	const src = `package p
func f() {
	type E enum { A }
	var outer E = A{}
	{
		type E enum { A }
		var inner E = A{}
		outer = inner
	}
}
`
	if _, err := checkEnumPackage(t, src); err == nil || !strings.Contains(err.Error(), "cannot use") {
		t.Fatalf("same-named local enum assignment error = %v", err)
	}
}

func TestImportedEnumSwitchVariantVisibility(t *testing.T) {
	pkg, err := checkEnumPackage(t, `package p; type E enum { Public; private }`)
	if err != nil {
		t.Fatal(err)
	}
	for _, switchStmt := range []string{
		`switch x { case private: default: }`,
		`switch x.(type) { case private: default: }`,
	} {
		fset := token.NewFileSet()
		file, err := parser.ParseFile(fset, "consumer.go", "package q; import \"p\"; func f(x p.E) { "+switchStmt+" }", parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		conf := types.Config{Importer: testImporter{"p": pkg}}
		if _, err := conf.Check("q", fset, []*ast.File{file}, nil); err == nil || !strings.Contains(err.Error(), "unexported enum variant") {
			t.Errorf("%s: visibility error = %v", switchStmt, err)
		}
	}
}

func TestImportedPrivateVariantSelectorVisibility(t *testing.T) {
	pkg, err := checkEnumPackage(t, `package p; type E enum { Public; private }`)
	if err != nil {
		t.Fatal(err)
	}
	for _, src := range []string{
		`package q; import "p"; type Alias = p.E; var _ = Alias.private{}`,
		`package q; import . "p"; var _ = E.private{}`,
	} {
		fset := token.NewFileSet()
		file, err := parser.ParseFile(fset, "consumer.go", src, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		conf := types.Config{Importer: testImporter{"p": pkg}}
		if _, err := conf.Check("q", fset, []*ast.File{file}, nil); err == nil || !strings.Contains(err.Error(), "unexported enum variant") {
			t.Errorf("%s: visibility error = %v", src, err)
		}
	}
}
