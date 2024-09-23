// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2_test

import (
	"fmt"
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
// the same Func Object as the original method. See also go.dev/issue/34421.
func TestEmbeddedMethod(t *testing.T) {
	const src = `package p; type I interface { error }`
	pkg := mustTypecheck(src, nil, nil)

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
	src   string
	obj   string
	want  string
	alias bool // needs materialized (and possibly generic) aliases
}{
	{"import \"io\"; var r io.Reader", "r", "var p.r io.Reader", false},

	{"const c = 1.2", "c", "const p.c untyped float", false},
	{"const c float64 = 3.14", "c", "const p.c float64", false},

	{"type t struct{f int}", "t", "type p.t struct{f int}", false},
	{"type t func(int)", "t", "type p.t func(int)", false},
	{"type t[P any] struct{f P}", "t", "type p.t[P any] struct{f P}", false},
	{"type t[P any] struct{f P}", "t.P", "type parameter P any", false},
	{"type C interface{m()}; type t[P C] struct{}", "t.P", "type parameter P p.C", false},

	{"type t = struct{f int}", "t", "type p.t = struct{f int}", false},
	{"type t = func(int)", "t", "type p.t = func(int)", false},
	{"type A = B; type B = int", "A", "type p.A = p.B", true},
	{"type A[P ~int] = struct{}", "A", "type p.A[P ~int] = struct{}", true}, // requires GOEXPERIMENT=aliastypeparams

	{"var v int", "v", "var p.v int", false},

	{"func f(int) string", "f", "func p.f(int) string", false},
	{"func g[P any](x P){}", "g", "func p.g[P any](x P)", false},
	{"func g[P interface{~int}](x P){}", "g.P", "type parameter P interface{~int}", false},
	{"", "any", "type any = interface{}", false},
}

func TestObjectString(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	for i, test := range testObjects {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			if test.alias {
				defer setGOEXPERIMENT("aliastypeparams")()
			}
			src := "package p; " + test.src
			conf := Config{Error: func(error) {}, Importer: defaultImporter(), EnableAlias: test.alias}
			pkg, err := typecheck(src, &conf, nil)
			if err != nil {
				t.Fatalf("%s: %s", src, err)
			}

			names := strings.Split(test.obj, ".")
			if len(names) != 1 && len(names) != 2 {
				t.Fatalf("%s: invalid object path %s", test.src, test.obj)
			}
			_, obj := pkg.Scope().LookupParent(names[0], nopos)
			if obj == nil {
				t.Fatalf("%s: %s not found", test.src, names[0])
			}
			if len(names) == 2 {
				if typ, ok := obj.Type().(interface{ TypeParams() *TypeParamList }); ok {
					obj = lookupTypeParamObj(typ.TypeParams(), names[1])
					if obj == nil {
						t.Fatalf("%s: %s not found", test.src, test.obj)
					}
				} else {
					t.Fatalf("%s: %s has no type parameters", test.src, names[0])
				}
			}

			if got := obj.String(); got != test.want {
				t.Errorf("%s: got %s, want %s", test.src, got, test.want)
			}
		})
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
