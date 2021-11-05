// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	"strings"
	"testing"
)

func parseSrc(path, src string) (*syntax.File, error) {
	return syntax.Parse(syntax.NewFileBase(path), strings.NewReader(src), nil, nil, 0)
}

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
		{NewTypeName(nopos, nil, "t2", &emptyInterface), true}, // type name refers to unnamed type
		{NewTypeName(nopos, pkg, "t3", n1), true},              // type name refers to named type with different type name
		{NewTypeName(nopos, nil, "t4", Typ[Int32]), true},      // type name refers to basic type with different name
		{NewTypeName(nopos, nil, "int32", Typ[Int32]), false},  // type name refers to basic type with same name
		{NewTypeName(nopos, pkg, "int32", Typ[Int32]), true},   // type name is declared in user-defined package (outside Universe)
		{NewTypeName(nopos, nil, "rune", Typ[Rune]), true},     // type name refers to basic type rune which is an alias already
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
