// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2_test

import (
	. "cmd/compile/internal/types2"
	"strings"
	"testing"
)

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
	pkg, err := typecheck(src, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	result := pkg.Scope().Lookup("Result").Type().(*Named)
	variants := result.EnumVariants()
	if len(variants) != 3 || variants[0].Obj().Name() != "Result.Ok" || variants[1].Obj().Name() != "Result.Err" || variants[2].Obj().Name() != "Result.None" {
		t.Fatalf("Result variants = %v, want [Result.Ok Result.Err Result.None]", variants)
	}
	ok, errVariant, none := variants[0], variants[1], variants[2]

	if _, ok := result.Underlying().(*Interface); !ok {
		t.Fatalf("Result underlying type is %T, want *Interface", result.Underlying())
	}
	if !Implements(result, result.Underlying().(*Interface)) {
		t.Error("Result does not implement its own enum interface")
	}
	for _, variant := range []*Named{ok, errVariant, none} {
		if _, ok := variant.Underlying().(*Struct); !ok {
			t.Errorf("%s underlying type is %T, want *Struct", variant.Obj().Name(), variant.Underlying())
		}
		if !AssignableTo(variant, result) {
			t.Errorf("%s is not assignable to Result", variant.Obj().Name())
		}
	}
	if fields := ok.Underlying().(*Struct); fields.NumFields() != 1 || fields.Field(0).Name() != "value" {
		t.Fatalf("Ok fields = %s, want value int", fields)
	}
	if method, _, _ := LookupFieldOrMethod(result, true, pkg, "Value"); method == nil {
		t.Error("method declared on enum Result was not collected")
	}
	if method, _, _ := LookupFieldOrMethod(ok, true, pkg, "Value"); method != nil {
		t.Error("enum method Value unexpectedly belongs to variant Ok")
	}
	if ok.EnumType() != result {
		t.Fatalf("Ok enum type = %v, want Result", ok.EnumType())
	}
	if AssignableTo(NewPointer(ok), result) {
		t.Error("*Ok is assignable to Result")
	}
	if Implements(NewPointer(ok), result.Underlying().(*Interface)) {
		t.Error("*Ok implements Result")
	}
	forged := NewNamed(NewTypeName(nopos, pkg, "Forged", nil), NewStruct([]*Var{
		NewField(nopos, pkg, "Ok", ok, true),
	}, nil), nil)
	if AssignableTo(forged, result) || Implements(forged, result.Underlying().(*Interface)) {
		t.Error("type embedding Ok implements Result")
	}
	orSig := NewSignatureType(nil, nil, nil,
		NewTuple(NewVar(nopos, nil, "", Typ[Int])),
		NewTuple(NewVar(nopos, nil, "", Typ[Int])), false)
	orIface := NewInterfaceType([]*Func{NewFunc(nopos, nil, "Value", orSig)}, nil)
	if Implements(result, orIface) {
		t.Error("enum convenience methods must not make Result implement another interface")
	}
}

func TestEnumVariantMethods(t *testing.T) {
	pkg, err := typecheck(`package p
type Decision enum { Allow; Deny { Reason string } }
var _ string = Decision.Allow{}.Variant()
func decisionVariant(decision Decision) string { return decision.Variant() }
`, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	decision := pkg.Scope().Lookup("Decision").Type().(*Named)
	for _, typ := range append([]*Named{decision}, decision.EnumVariants()...) {
		method, _, _ := LookupFieldOrMethod(typ, true, pkg, "Variant")
		fn, ok := method.(*Func)
		if !ok {
			t.Fatalf("%s Variant method = %T, want *Func", typ, method)
		}
		sig := fn.Type().(*Signature)
		if sig.Params().Len() != 0 || sig.Results().Len() != 1 || sig.Results().At(0).Type() != Typ[String] {
			t.Fatalf("%s Variant signature = %s, want func() string", typ, sig)
		}
	}
}

func TestEnumVariantMethodCannotBeOverridden(t *testing.T) {
	_, err := typecheck(`package p
type Decision enum { Allow }
func (Decision) Variant() string { return "custom" }
`, nil, nil)
	if err == nil || !strings.Contains(err.Error(), "conflicts with generated enum method Variant") {
		t.Fatalf("Variant override error = %v", err)
	}
}

func TestVariantMethodOnRecursiveNonEnum(t *testing.T) {
	_, err := typecheck(`package p
type A struct { B *B }
type B A
func (B) Variant() string { return "B" }
`, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
}

func TestEnumVersion(t *testing.T) {
	conf := Config{GoVersion: "go1.27"}
	_, err := typecheck("package p; type E enum { A }", &conf, nil)
	if err == nil || !strings.Contains(err.Error(), "requires go1.28 or later") {
		t.Fatalf("enum version error = %v", err)
	}
}

func TestEnumVariantTypeRejected(t *testing.T) {
	const src = `package p
type Result enum { Ok { value int }; Err }
func (o Result.Ok) Value() int { return o.value }
`
	_, err := typecheck(src, nil, nil)
	if err == nil || !strings.Contains(err.Error(), "is a constructor, not a type") {
		t.Fatalf("variant type error = %v", err)
	}
}

func TestEnumMethodVariantCollisionRejected(t *testing.T) {
	const src = `package p
type Result enum { Ok }
func (Result) Ok() {}
`
	_, err := typecheck(src, nil, nil)
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
	if _, err := typecheck(exhaustive, nil, nil); err != nil {
		t.Fatal(err)
	}

	const missing = `package p
type Result enum { Ok; Err; None }
func inspect(r Result) { switch r.(type) { case Ok: } }
`
	_, err := typecheck(missing, nil, nil)
	if err == nil || !strings.Contains(err.Error(), "non-exhaustive enum switch on Result; missing Err, None, nil") {
		t.Fatalf("non-exhaustive switch error = %v", err)
	}

	const withDefault = `package p
type Result enum { Ok; Err }
func inspect(r Result) { switch r.(type) { case Ok:; default: } }
`
	if _, err := typecheck(withDefault, nil, nil); err != nil {
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
	if _, err := typecheck(exhaustive, nil, nil); err != nil {
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
	if _, err := typecheck(expressions, nil, nil); err != nil {
		t.Fatal(err)
	}

	const duplicate = `package p
type Result enum { Ok; Err }
func inspect(result Result) { switch result { case Ok:; case Ok:; case Err:; case nil: } }
`
	_, err := typecheck(duplicate, nil, nil)
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
			if _, err := typecheck(src, nil, nil); err != nil {
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
	_, err := typecheck(src, nil, nil)
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
		_, err := typecheck(src, nil, nil)
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
	if _, err := typecheck(valid, nil, nil); err != nil {
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
		_, err := typecheck(src, nil, nil)
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
	pkg, err := typecheck(src, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	option := pkg.Scope().Lookup("Option").Type().(*Named)
	variants := option.EnumVariants()
	some, none := variants[0], variants[1]
	optionParam := option.TypeParams().At(0)
	someParam := some.TypeParams().At(0)
	noneParam := none.TypeParams().At(0)
	if optionParam == someParam || optionParam == noneParam || someParam == noneParam {
		t.Fatal("generic enum and variants share type parameter objects")
	}
	if fieldType := some.Underlying().(*Struct).Field(0).Type(); fieldType != someParam {
		t.Fatalf("Some.value type = %v, want Some type parameter %v", fieldType, someParam)
	}
	if method, _, _ := LookupFieldOrMethod(option, true, pkg, "Or"); method == nil {
		t.Fatal("generic enum method Or was not collected")
	}
	marker := some.Method(0).Type().(*Signature)
	recv := marker.Recv().Type().(*Named)
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
	if _, err := typecheck(src, nil, nil); err != nil {
		t.Fatalf("contextual variants: %v", err)
	}

	const invalid = `package p
type Result enum { Ok }
var _ = Ok{}
`
	_, err := typecheck(invalid, nil, nil)
	if err == nil || !strings.Contains(err.Error(), "undefined: Ok") {
		t.Fatalf("bare variant error = %v", err)
	}
}

func TestDuplicateEnumVariant(t *testing.T) {
	const src = `package p
type E enum { A; A }
`
	_, err := typecheck(src, nil, nil)
	if err == nil || !strings.Contains(err.Error(), "A redeclared") {
		t.Fatalf("duplicate variant error = %v, want redeclaration error", err)
	}
}

func TestEnumVariantVisibility(t *testing.T) {
	const src = `package p
type E enum { Public; private }
`
	pkg, err := typecheck(src, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	variants := pkg.Scope().Lookup("E").Type().(*Named).EnumVariants()
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
	if _, err := typecheck(src, nil, nil); err == nil || !strings.Contains(err.Error(), "cannot use") {
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
	if _, err := typecheck(src, nil, nil); err == nil || !strings.Contains(err.Error(), "cannot use") {
		t.Fatalf("same-named local enum assignment error = %v", err)
	}
}

func TestImportedEnumSwitchVariantVisibility(t *testing.T) {
	pkg, err := typecheck(`package p; type E enum { Public; private }`, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, switchStmt := range []string{
		`switch x { case private: default: }`,
		`switch x.(type) { case private: default: }`,
	} {
		conf := Config{Importer: testImporter{"p": pkg}}
		_, err := typecheck("package q; import \"p\"; func f(x p.E) { "+switchStmt+" }", &conf, nil)
		if err == nil || !strings.Contains(err.Error(), "unexported enum variant") {
			t.Errorf("%s: visibility error = %v", switchStmt, err)
		}
	}
}

func TestImportedPrivateVariantSelectorVisibility(t *testing.T) {
	pkg, err := typecheck(`package p; type E enum { Public; private }`, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, src := range []string{
		`package q; import "p"; type Alias = p.E; var _ = Alias.private{}`,
		`package q; import . "p"; var _ = E.private{}`,
	} {
		conf := Config{Importer: testImporter{"p": pkg}}
		if _, err := typecheck(src, &conf, nil); err == nil || !strings.Contains(err.Error(), "unexported enum variant") {
			t.Errorf("%s: visibility error = %v", src, err)
		}
	}
}
