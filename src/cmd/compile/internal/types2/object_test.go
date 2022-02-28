// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2_test

import (
	"cmd/compile/internal/syntax"
	"internal/testenv"
	"strings"
	"testing"

	. "cmd/compile/internal/types2"
)

func TestIsAlias(t *testing.T) {
	check := func(obj *TypeName, want bool) {
		if got := obj.IsAlias(); got != want {
			t.Errorf("%v: got IsAlias = %v; want %v", obj, got, want)
		}
	}

	// predeclared types
	check(Unsafe.Scope().Lookup("Pointer").(*TypeName), false)
	for _, name := range Universe.Names() {
		if obj, _ := Universe.Lookup(name).(*TypeName); obj != nil {
			check(obj, name == "any" || name == "byte" || name == "rune")
		}
	}

	// various other types
	pkg := NewPackage("p", "p")
	t1 := NewTypeName(nopos, pkg, "t1", nil)
	n1 := NewNamed(t1, new(Struct), nil)
	t5 := NewTypeName(nopos, pkg, "t5", nil)
	NewTypeParam(t5, nil)
	for _, test := range []struct {
		name  *TypeName
		alias bool
	}{
		{NewTypeName(nopos, nil, "t0", nil), false}, // no type yet
		{NewTypeName(nopos, pkg, "t0", nil), false}, // no type yet
		{t1, false}, // type name refers to named type and vice versa
		{NewTypeName(nopos, nil, "t2", NewInterfaceType(nil, nil)), true}, // type name refers to unnamed type
		{NewTypeName(nopos, pkg, "t3", n1), true},                         // type name refers to named type with different type name
		{NewTypeName(nopos, nil, "t4", Typ[Int32]), true},                 // type name refers to basic type with different name
		{NewTypeName(nopos, nil, "int32", Typ[Int32]), false},             // type name refers to basic type with same name
		{NewTypeName(nopos, pkg, "int32", Typ[Int32]), true},              // type name is declared in user-defined package (outside Universe)
		{NewTypeName(nopos, nil, "rune", Typ[Rune]), true},                // type name refers to basic type rune which is an alias already
		{t5, false}, // type name refers to type parameter and vice versa
	} {
		check(test.name, test.alias)
	}
}

// TestEmbeddedMethod checks that an embedded method is represented by
// the same Func Object as the original method. See also issue #34421.
func TestEmbeddedMethod(t *testing.T) {
	const src = `package p; type I interface { error }`

	// type-check src
	f, err := parseSrc("", src)
	if err != nil {
		t.Fatalf("parse failed: %s", err)
	}
	var conf Config
	pkg, err := conf.Check(f.PkgName.Value, []*syntax.File{f}, nil)
	if err != nil {
		t.Fatalf("typecheck failed: %s", err)
	}

	// get original error.Error method
	eface := Universe.Lookup("error")
	orig, _, _ := LookupFieldOrMethod(eface.Type(), false, nil, "Error")
	if orig == nil {
		t.Fatalf("original error.Error not found")
	}

	// get embedded error.Error method
	iface := pkg.Scope().Lookup("I")
	embed, _, _ := LookupFieldOrMethod(iface.Type(), false, nil, "Error")
	if embed == nil {
		t.Fatalf("embedded error.Error not found")
	}

	// original and embedded Error object should be identical
	if orig != embed {
		t.Fatalf("%s (%p) != %s (%p)", orig, orig, embed, embed)
	}
}

var testObjects = []struct {
	src  string
	obj  string
	want string
}{
	{"import \"io\"; var r io.Reader", "r", "var p.r io.Reader"},

	{"const c = 1.2", "c", "const p.c untyped float"},
	{"const c float64 = 3.14", "c", "const p.c float64"},

	{"type t struct{f int}", "t", "type p.t struct{f int}"},
	{"type t func(int)", "t", "type p.t func(int)"},
	{"type t[P any] struct{f P}", "t", "type p.t[P any] struct{f P}"},
	{"type t[P any] struct{f P}", "t.P", "type parameter P any"},
	{"type C interface{m()}; type t[P C] struct{}", "t.P", "type parameter P p.C"},

	{"type t = struct{f int}", "t", "type p.t = struct{f int}"},
	{"type t = func(int)", "t", "type p.t = func(int)"},

	{"var v int", "v", "var p.v int"},

	{"func f(int) string", "f", "func p.f(int) string"},
	{"func g[P any](x P){}", "g", "func p.g[P any](x P)"},
	{"func g[P interface{~int}](x P){}", "g.P", "type parameter P interface{~int}"},
	{"", "any", "type any = interface{}"},
}

func TestObjectString(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	for _, test := range testObjects {
		src := "package p; " + test.src
		pkg, err := makePkg(src)
		if err != nil {
			t.Errorf("%s: %s", src, err)
			continue
		}

		names := strings.Split(test.obj, ".")
		if len(names) != 1 && len(names) != 2 {
			t.Errorf("%s: invalid object path %s", test.src, test.obj)
			continue
		}
		_, obj := pkg.Scope().LookupParent(names[0], nopos)
		if obj == nil {
			t.Errorf("%s: %s not found", test.src, names[0])
			continue
		}
		if len(names) == 2 {
			if typ, ok := obj.Type().(interface{ TypeParams() *TypeParamList }); ok {
				obj = lookupTypeParamObj(typ.TypeParams(), names[1])
				if obj == nil {
					t.Errorf("%s: %s not found", test.src, test.obj)
					continue
				}
			} else {
				t.Errorf("%s: %s has no type parameters", test.src, names[0])
				continue
			}
		}

		if got := obj.String(); got != test.want {
			t.Errorf("%s: got %s, want %s", test.src, got, test.want)
		}
	}
}

func lookupTypeParamObj(list *TypeParamList, name string) Object {
	for i := 0; i < list.Len(); i++ {
		tpar := list.At(i)
		if tpar.Obj().Name() == name {
			return tpar.Obj()
		}
	}
	return nil
}
