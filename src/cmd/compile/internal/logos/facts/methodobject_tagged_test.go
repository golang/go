// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package facts_test

import (
	"reflect"
	"testing"

	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"cmd/internal/goobj"
)

func TestTaggedMethodObjectConstructors(t *testing.T) {
	for _, names := range []struct{ typeName, value, pointer string }{
		{"Box", "Box.Read", "(*Box).Read"},
		{"Box@simd0", "Box@simd0.Read", "(*Box@simd0).Read"},
		{"Box@simd128", "Box@simd128.Read", "(*Box@simd128).Read"},
		{"Ω@simd128", "Ω@simd128.Read", "(*Ω@simd128).Read"},
	} {
		t.Run(names.typeName, func(t *testing.T) {
			pkg := types2.NewPackage("example.com/methodconstructors", "methodconstructors")
			pos := syntax.Pos{}
			named := types2.NewNamed(types2.NewTypeName(pos, pkg, names.typeName, nil), types2.NewStruct(nil, nil), nil)
			pointer := types2.NewPointer(named)
			alias := types2.NewAlias(types2.NewTypeName(pos, pkg, "Alias", nil), named)
			pointerAlias := types2.NewAlias(types2.NewTypeName(pos, pkg, "PointerAlias", nil), pointer)
			for _, tc := range []struct {
				name       string
				recv       types2.Type
				want       string
				objectWant string
			}{
				{"value", named, names.value, names.value},
				{"pointer", pointer, names.pointer, names.pointer},
				{"alias value", alias, names.value, names.value},
				{"alias pointer", pointerAlias, names.pointer, names.pointer},
				{"const value", &types2.ConstView{Elem: named}, names.value, ""},
				{"const pointer", &types2.ConstView{Elem: pointer}, names.pointer, ""},
				{"pointer to const", types2.NewPointer(&types2.ConstView{Elem: named}), names.pointer, ""},
				{"const alias pointer", &types2.ConstView{Elem: pointerAlias}, names.pointer, ""},
			} {
				t.Run(tc.name, func(t *testing.T) {
					recv := types2.NewVar(pos, pkg, "self", tc.recv)
					fn := types2.NewFunc(pos, pkg, "Read", types2.NewSignatureType(recv, nil, nil, nil, nil, false))
					if got := facts.MethodObject(names.typeName, recv, fn.Name()); got != tc.want {
						t.Errorf("MethodObject = %q, want %q", got, tc.want)
					}
					// ObjectName does not currently unwrap ConstView. Preserve that
					// separate contract; this codec slice does not change it.
					if got := facts.ObjectName(fn); got != tc.objectWant {
						t.Errorf("ObjectName = %q, want %q", got, tc.objectWant)
					}
					fp := goobj.FingerprintType{1, 2, 3, 4, 5, 6, 7, 8}
					in := &facts.Facts{Fingerprint: fp, Fns: []facts.FnFact{{
						Object: facts.MethodObject(names.typeName, recv, fn.Name()), Readonly: true, ParamMutated: "0",
					}}}
					out, err := facts.Decode(facts.Encode(fp, in))
					if err != nil {
						t.Fatalf("Decode(Encode(MethodObject)): %v", err)
					}
					if !reflect.DeepEqual(out, in) {
						t.Fatalf("method fact round trip = %+v, want %+v", out, in)
					}
					if out.Fns[0].Object != tc.want || out.Fns[0].ParamMutated != "0" {
						t.Fatalf("round trip lost the canonical key or known-zero mask: %+v", out.Fns[0])
					}
				})
			}
			if got := facts.MethodObject(names.typeName, nil, "Read"); got != names.value {
				t.Errorf("nil receiver method key = %q, want %q", got, names.value)
			}
		})
	}
}

func TestTaggedMethodCodecGrammar(t *testing.T) {
	for _, tc := range []struct {
		object string
		valid  bool
	}{
		{"Box.Read", true},
		{"(*Box).Read", true},
		{"Box@simd128", true},
		{"(*Box)@simd128", true},
		{"(Box).Read@simd128", true},
		{"Box.Read@simd128", true},
		{"(*Box).Read@simd128", true},
		{"Box@simd128.Read", true},
		{"(*Box@simd128).Read", true},
		{"Box@simd0.Read", true},
		{"Ω@simd128.Read", true},
		{"Box@.Read", false},
		{"Box@1.Read", false},
		{"Box@simd128!.Read", false},
		{"Box@@simd128.Read", false},
		{"Box@simd128@again.Read", false},
		{"Box@simd128.Read@again", false},
		{"(*Box@simd128).Read@again", false},
		{"(*Box@).Read", false},
		{"(*Box@simd128.Read", false},
		{"Box@simd128).Read", false},
		{"Box@simd128..Read", false},
		{"Box@simd128.Read.More", false},
		{"local·1@simd128.Read", false},
		{"Box[go.shape.int]@simd128.Read", false},
		{"Box.Read@", false},
		{"Box.Read@1", false},
		{"Box.Read@simd128!", false},
		{"Box.Read@simd128@again", false},
		{"Box..Read@simd128", false},
		{"Box.Read.More@simd128", false},
		{"local·1.Read@simd128", false},
		{"Box[go.shape.int].Read@simd128", false},
	} {
		t.Run(tc.object, func(t *testing.T) {
			if got := facts.ValidObject(tc.object); got != tc.valid {
				t.Fatalf("ValidObject = %v, want %v", got, tc.valid)
			}
			wire := "logos-facts 3\nfingerprint 0102030405060708\nfn " + tc.object + " ParamMutated=0\n"
			out, err := facts.Decode([]byte(wire))
			if !tc.valid {
				if err == nil {
					t.Fatal("decoder accepted an unsupported method spelling")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(out.Fns) != 1 || out.Fns[0].Object != tc.object || out.Fns[0].ParamMutated != "0" {
				t.Fatalf("decoder changed the canonical method fact: %+v", out.Fns)
			}
		})
	}
}

func TestTaggedInterfaceReadonlyFactWriter(t *testing.T) {
	pkg := types2.NewPackage("example.com/interfacemethodwriter", "interfacemethodwriter")
	pos := syntax.Pos{}
	read := types2.NewFunc(pos, pkg, "Read", types2.NewSignatureType(nil, nil, nil, nil, nil, false))
	read.SetLogosReadonly(true)
	named := types2.NewNamed(types2.NewTypeName(pos, pkg, "Reader@simd128", nil), types2.NewInterfaceType([]*types2.Func{read}, nil), nil)
	pkg.Scope().Insert(named.Obj())
	in := facts.Write(pkg, &types2.Info{}, nil, nil)
	if len(in.Fns) != 1 || in.Fns[0].Object != "Reader@simd128.Read" || !in.Fns[0].Readonly {
		t.Fatalf("readonly interface writer = %+v, want Reader@simd128.Read readonly fact", in.Fns)
	}
	fp := goobj.FingerprintType{1, 2, 3, 4, 5, 6, 7, 8}
	out, err := facts.Decode(facts.Encode(fp, in))
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Fns) != 1 || out.Fns[0].Object != "Reader@simd128.Read" || !out.Fns[0].Readonly {
		t.Fatalf("readonly interface round trip lost the method fact: %+v", out.Fns)
	}
}
