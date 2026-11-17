// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"testing"

	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"cmd/internal/goobj"
)

func taggedMethodFixture(t *testing.T, pointer bool) (*types2.Package, *types2.Named, *types2.Func, string) {
	t.Helper()
	pkg := types2.NewPackage("example.com/"+t.Name(), "taggedmethod")
	pos := syntax.Pos{}
	named := types2.NewNamed(types2.NewTypeName(pos, pkg, "Box@simd128", nil), types2.NewStruct(nil, nil), nil)
	var receiver types2.Type = named
	object := "Box@simd128.Read"
	if pointer {
		receiver = types2.NewPointer(named)
		object = "(*Box@simd128).Read"
	}
	params := types2.NewTuple(
		types2.NewParam(pos, pkg, "value", types2.NewPointer(types2.Typ[types2.Int])),
		types2.NewParam(pos, pkg, "count", types2.Typ[types2.Int]),
	)
	results := types2.NewTuple(types2.NewVar(pos, pkg, "", types2.NewSlice(types2.Typ[types2.Int])))
	method := types2.NewFunc(pos, pkg, "Read", types2.NewSignatureType(types2.NewVar(pos, pkg, "self", receiver), nil, nil, params, results, false))
	named.AddMethod(method)
	pkg.Scope().Insert(named.Obj())
	pkg.Scope().InsertLazy(object, func() types2.Object {
		t.Fatalf("resolved method symbol %q through the package scope", object)
		return nil
	})
	return pkg, named, method, object
}

func TestTaggedMethodFactLookup(t *testing.T) {
	for _, tc := range []struct {
		name    string
		pointer bool
	}{
		{"value", false},
		{"pointer", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pkg, named, method, object := taggedMethodFixture(t, tc.pointer)
			before := method.Type()
			if _, _, known := method.LogosAssertEffects(); known || method.LogosReadonly() {
				t.Fatal("new method already has trusted metadata")
			}
			fp := goobj.FingerprintType{1, 2, 3, 4, 5, 6, 7, 8}
			decoded, err := facts.Decode(facts.Encode(fp, &facts.Facts{Fns: []facts.FnFact{{
				Object: object, Readonly: true, HasAssert: true, AssertTP: "0", AssertRTP: "0",
			}}}))
			if err != nil {
				t.Fatal(err)
			}
			if err := ApplyFuncFacts(pkg, decoded.Fns); err != nil {
				t.Fatal(err)
			}
			if !method.LogosReadonly() {
				t.Fatal("canonical method fact did not reach the exact tagged receiver method")
			}
			checkTaggedFactMasks(t, method, "0", "0")
			if pkg.Scope().Lookup(named.Obj().Name()) != named.Obj() || named.Method(0) != method || method.Type() != before {
				t.Fatal("method metadata application replaced a canonical declaration or signature")
			}
		})
	}
}

func TestTaggedMethodFactRejectsInvalidMask(t *testing.T) {
	for _, pointer := range []bool{false, true} {
		name := "value"
		if pointer {
			name = "pointer"
		}
		t.Run(name, func(t *testing.T) {
			pkg, _, method, object := taggedMethodFixture(t, pointer)
			err := ApplyFuncFacts(pkg, []facts.FnFact{{Object: object, Readonly: true, HasAssert: true, AssertTP: "1", AssertRTP: "0"}})
			if err == nil {
				t.Fatal("canonical method lookup silently accepted an assertion mask outside declaration arity")
			}
			if _, _, known := method.LogosAssertEffects(); known || method.LogosReadonly() {
				t.Fatal("rejected method record published metadata")
			}
		})
	}
}

// This named control must fail when only optfacts.go is restored from the
// frozen pre-repair file after the codec accepts receiver-local tags: the old
// package-name guard then admits this lazy method symbol.
func TestTaggedInterfaceMethodFactLookup(t *testing.T) {
	pkg := types2.NewPackage("example.com/taggedinterfacelookup", "taggedinterfacelookup")
	pos := syntax.Pos{}
	read := types2.NewFunc(pos, pkg, "Read", types2.NewSignatureType(nil, nil, nil, nil, nil, false))
	named := types2.NewNamed(types2.NewTypeName(pos, pkg, "Reader@simd128", nil), types2.NewInterfaceType([]*types2.Func{read}, nil), nil)
	pkg.Scope().Insert(named.Obj())
	pkg.Scope().InsertLazy("Reader@simd128.Read", func() types2.Object {
		t.Fatal("resolved an interface method symbol through the package scope")
		return nil
	})
	if err := ApplyFuncFacts(pkg, []facts.FnFact{{Object: "Reader@simd128.Read", Readonly: true}}); err != nil {
		t.Fatal(err)
	}
	if !read.LogosReadonly() {
		t.Fatal("canonical tagged interface method fact was silently ignored")
	}
	if pkg.Scope().Lookup(named.Obj().Name()) != named.Obj() || named.Underlying().(*types2.Interface).ExplicitMethod(0) != read {
		t.Fatal("interface metadata application replaced a canonical object")
	}
}

func TestTaggedMethodStructuralFactLookup(t *testing.T) {
	for _, pointer := range []bool{false, true} {
		name := "value"
		if pointer {
			name = "pointer"
		}
		t.Run(name, func(t *testing.T) {
			pkg, original, method, object := taggedMethodFixture(t, pointer)
			out, err := CloneWithOverlaysAndReadonly(pkg, nil,
				[]facts.PathFact{{Object: object, Path: "R0"}},
				[]facts.FnFact{{Object: object, Readonly: true, HasAssert: true, AssertTP: "0", AssertRTP: "0"}}, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			if out == pkg {
				t.Fatal("structural method fact did not clone the package")
			}
			obj := out.Scope().Lookup(original.Obj().Name())
			if obj == nil {
				t.Fatal("structural clone omitted the tagged receiver type")
			}
			cloned, ok := obj.Type().(*types2.Named)
			if !ok || cloned == original || cloned.NumMethods() != 1 {
				t.Fatalf("structural clone lost the receiver or method: %v", obj)
			}
			got := cloned.Method(0)
			if _, ok := got.Signature().Results().At(0).Type().(*types2.ConstView); !ok {
				t.Fatalf("canonical method const result overlay was lost: %v", got.Signature())
			}
			if !got.LogosReadonly() {
				t.Fatal("structural method clone lost readonly metadata")
			}
			checkTaggedFactMasks(t, got, "0", "0")
			if _, ok := method.Signature().Results().At(0).Type().(*types2.Slice); !ok {
				t.Fatal("structural method overlay modified the original result")
			}
			if _, _, known := method.LogosAssertEffects(); known || method.LogosReadonly() {
				t.Fatal("structural method overlay published metadata into the original")
			}
		})
	}
}

func TestTaggedMethodOptionFactLookup(t *testing.T) {
	for _, pointer := range []bool{false, true} {
		name := "value"
		if pointer {
			name = "pointer"
		}
		t.Run(name, func(t *testing.T) {
			pkg, _, method, object := taggedMethodFixture(t, pointer)
			fp := goobj.FingerprintType{1, 2, 3, 4, 5, 6, 7, 8}
			in := &facts.Facts{Opts: []facts.PathFact{
				{Object: object, Path: "P0"},
				{Object: "Other@simd128.Read", Path: "P1"},
			}}
			if _, err := facts.Import(pkg.Path(), string(fp[:]), facts.Encode(fp, in)); err != nil {
				t.Fatal(err)
			}
			got := optParamSet(method)
			if len(got) != 1 || !got[0] || got[1] {
				t.Fatalf("canonical method option parameters = %v, want only parameter 0", got)
			}
		})
	}
}

func TestTaggedMethodIdentityRemainsDistinct(t *testing.T) {
	pkg := types2.NewPackage("example.com/distinctmethodtags", "distinctmethodtags")
	pos := syntax.Pos{}
	makeMethod := func(typeName, methodName string) *types2.Func {
		named := types2.NewNamed(types2.NewTypeName(pos, pkg, typeName, nil), types2.NewStruct(nil, nil), nil)
		recv := types2.NewVar(pos, pkg, "self", named)
		fn := types2.NewFunc(pos, pkg, methodName, types2.NewSignatureType(recv, nil, nil, nil, nil, false))
		named.AddMethod(fn)
		pkg.Scope().Insert(named.Obj())
		return fn
	}
	receiverTag := makeMethod("Box@simd128", "Read")
	methodTag := makeMethod("Box", "Read@simd128")
	fp := goobj.FingerprintType{1, 2, 3, 4, 5, 6, 7, 8}
	decoded, err := facts.Decode(facts.Encode(fp, &facts.Facts{Fns: []facts.FnFact{
		{Object: "Box@simd128.Read", Readonly: true},
		{Object: "Box.Read@simd128", HasAssert: true, AssertTP: "0", AssertRTP: "0"},
	}}))
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded.Fns) != 2 {
		t.Fatalf("codec collapsed distinct tagged declarations: %+v", decoded.Fns)
	}
	if err := ApplyFuncFacts(pkg, decoded.Fns); err != nil {
		t.Fatal(err)
	}
	if !receiverTag.LogosReadonly() || methodTag.LogosReadonly() {
		t.Fatal("receiver-local readonly metadata crossed into the method-local tag identity")
	}
	if _, _, known := receiverTag.LogosAssertEffects(); known {
		t.Fatal("method-local assertion metadata crossed into the receiver-local tag identity")
	}
	checkTaggedFactMasks(t, methodTag, "0", "0")
}
