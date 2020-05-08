// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
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
			check(obj, name == "byte" || name == "rune")
		}
	}

	// various other types
	pkg := NewPackage("p", "p")
	t1 := NewTypeName(0, pkg, "t1", nil)
	n1 := NewNamed(t1, new(Struct), nil)
	for _, test := range []struct {
		name  *TypeName
		alias bool
	}{
		{NewTypeName(0, nil, "t0", nil), false},            // no type yet
		{NewTypeName(0, pkg, "t0", nil), false},            // no type yet
		{t1, false},                                        // type name refers to named type and vice versa
		{NewTypeName(0, nil, "t2", &emptyInterface), true}, // type name refers to unnamed type
		{NewTypeName(0, pkg, "t3", n1), true},              // type name refers to named type with different type name
		{NewTypeName(0, nil, "t4", Typ[Int32]), true},      // type name refers to basic type with different name
		{NewTypeName(0, nil, "int32", Typ[Int32]), false},  // type name refers to basic type with same name
		{NewTypeName(0, pkg, "int32", Typ[Int32]), true},   // type name is declared in user-defined package (outside Universe)
		{NewTypeName(0, nil, "rune", Typ[Rune]), true},     // type name refers to basic type rune which is an alias already
	} {
		check(test.name, test.alias)
	}
}

// TestEmbeddedMethod checks that an embedded method is represented by
// the same Func Object as the original method. See also issue #34421.
func TestEmbeddedMethod(t *testing.T) {
	const src = `package p; type I interface { error }`

	// type-check src
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", src, 0)
	if err != nil {
		t.Fatalf("parse failed: %s", err)
	}
	var conf Config
	pkg, err := conf.Check(f.Name.Name, fset, []*ast.File{f}, nil)
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
