// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"testing"

	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func taggedFactFunction(pkg *types2.Package, name string) *types2.Func {
	pos := syntax.Pos{}
	param := types2.NewTypeParam(types2.NewTypeName(pos, pkg, "T", nil), types2.NewInterfaceType(nil, nil))
	fn := types2.NewFunc(pos, pkg, name, types2.NewSignatureType(nil, nil, []*types2.TypeParam{param}, nil, nil, false))
	pkg.Scope().Insert(fn)
	return fn
}

func checkTaggedFactMasks(t *testing.T, fn *types2.Func, wantTP, wantRTP string) {
	t.Helper()
	tp, rtp, known := fn.LogosAssertEffects()
	if !known || types2.LogosAssertMaskHex(tp) != wantTP || types2.LogosAssertMaskHex(rtp) != wantRTP {
		t.Errorf("%s: assertion metadata known=%v TP=%q RTP=%q, want true/%q/%q", fn.Name(), known,
			types2.LogosAssertMaskHex(tp), types2.LogosAssertMaskHex(rtp), wantTP, wantRTP)
	}
}

func TestFactScopeTaggedFunctionAssertions(t *testing.T) {
	for _, name := range []string{"Project", "Project@simd0", "Project@simd128", "Project@simd256", "Project@simd512", "Ω@simd128", "_value2@simd128"} {
		for _, mask := range []string{"0", "1"} {
			t.Run(name+"/mask="+mask, func(t *testing.T) {
				pkg := types2.NewPackage("example.com/taggedfacts", "taggedfacts")
				fn := taggedFactFunction(pkg, name)
				if _, _, known := fn.LogosAssertEffects(); known {
					t.Fatal("new function already has trusted assertion metadata")
				}
				before := fn.Type()
				record := facts.FnFact{Object: name, HasAssert: true, AssertTP: mask, AssertRTP: "0"}
				if err := ApplyFuncFacts(pkg, []facts.FnFact{record}); err != nil {
					t.Fatal(err)
				}
				if pkg.Scope().Lookup(name) != fn || fn.Type() != before {
					t.Fatal("metadata application replaced the canonical declaration or signature")
				}
				checkTaggedFactMasks(t, fn, mask, "0")
			})
		}
	}
}

func TestFactScopeTaggedFunctionRejectsInvalidMasks(t *testing.T) {
	for _, tc := range []struct{ name, tp, rtp string }{
		{"outside function arity", "2", "0"},
		{"receiver bit on function", "0", "1"},
		{"missing receiver mask", "0", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pkg := types2.NewPackage("example.com/taggedfacts", "taggedfacts")
			fn := taggedFactFunction(pkg, "Project@simd128")
			record := facts.FnFact{Object: fn.Name(), HasAssert: true, AssertTP: tc.tp, AssertRTP: tc.rtp, Readonly: true}
			if err := ApplyFuncFacts(pkg, []facts.FnFact{record}); err == nil {
				t.Fatal("invalid tagged assertion record was accepted")
			}
			if _, _, known := fn.LogosAssertEffects(); known || fn.LogosReadonly() {
				t.Fatal("invalid tagged assertion record published metadata")
			}
		})
	}
}

func TestFactScopeTaggedReadonlyType(t *testing.T) {
	for _, path := range []string{"MRead", "FView.MRead"} {
		t.Run(path, func(t *testing.T) {
			pkg := types2.NewPackage("example.com/taggedfacts", "taggedfacts")
			pos := syntax.Pos{}
			read := types2.NewFunc(pos, pkg, "Read", types2.NewSignatureType(nil, nil, nil, nil, nil, false))
			var underlying types2.Type = types2.NewInterfaceType([]*types2.Func{read}, nil)
			if path == "FView.MRead" {
				underlying = types2.NewStruct([]*types2.Var{types2.NewField(pos, pkg, "View", underlying, false)}, nil)
			}
			named := types2.NewNamed(types2.NewTypeName(pos, pkg, "Shelf@simd128", nil), underlying, nil)
			pkg.Scope().Insert(named.Obj())
			if err := ApplyReadonlyFactsInPlace(pkg, []facts.ReadonlyFact{{Object: named.Obj().Name(), Path: path}}); err != nil {
				t.Fatal(err)
			}
			if !read.LogosReadonly() {
				t.Fatal("tagged type root lost reachable interface method metadata")
			}
			if pkg.Scope().Lookup(named.Obj().Name()) != named.Obj() || named.Underlying() != underlying {
				t.Fatal("readonly metadata application replaced the canonical type")
			}
		})
	}
}

func TestFactScopeTaggedReachableMethod(t *testing.T) {
	pkg := types2.NewPackage("example.com/taggedfacts", "taggedfacts")
	pos := syntax.Pos{}
	constraint := types2.NewInterfaceType(nil, nil)
	param := types2.NewTypeParam(types2.NewTypeName(pos, pkg, "T", nil), constraint)
	hidden := types2.NewNamed(types2.NewTypeName(pos, pkg, "hidden", nil), types2.NewStruct(nil, nil), nil)
	hidden.SetTypeParams([]*types2.TypeParam{param})
	receiverParam := types2.NewTypeParam(types2.NewTypeName(pos, pkg, "R", nil), constraint)
	receiverType, err := types2.Instantiate(nil, hidden, []types2.Type{receiverParam}, false)
	if err != nil {
		t.Fatal(err)
	}
	receiver := types2.NewVar(pos, pkg, "self", receiverType)
	method := types2.NewFunc(pos, pkg, "As", types2.NewSignatureType(receiver, []*types2.TypeParam{receiverParam}, nil, nil, nil, false))
	hidden.AddMethod(method)
	instance, err := types2.Instantiate(nil, hidden, []types2.Type{types2.Typ[types2.Int]}, false)
	if err != nil {
		t.Fatal(err)
	}
	root := types2.NewNamed(types2.NewTypeName(pos, pkg, "Shelf@simd128", nil),
		types2.NewStruct([]*types2.Var{types2.NewField(pos, pkg, "Value", instance, false)}, nil), nil)
	pkg.Scope().Insert(root.Obj())
	if pkg.Scope().Lookup("hidden") != nil {
		t.Fatal("method owner must be reachable only through the tagged root")
	}
	if err := ApplyFuncFacts(pkg, []facts.FnFact{{Object: "hidden.As", HasAssert: true, AssertTP: "0", AssertRTP: "1", Readonly: true}}); err != nil {
		t.Fatal(err)
	}
	checkTaggedFactMasks(t, method, "0", "1")
	if !method.LogosReadonly() {
		t.Fatal("tagged declaration traversal lost reachable method metadata")
	}
}

func TestFactScopeTaggedClones(t *testing.T) {
	for _, structural := range []bool{false, true} {
		name := "declaration"
		if structural {
			name = "structural"
		}
		t.Run(name, func(t *testing.T) {
			pkg := types2.NewPackage("example.com/taggedfacts", "taggedfacts")
			pos := syntax.Pos{}
			fn := taggedFactFunction(pkg, "Project@simd128")
			read := types2.NewFunc(pos, pkg, "Read", types2.NewSignatureType(nil, nil, nil, nil, nil, false))
			shelf := types2.NewNamed(types2.NewTypeName(pos, pkg, "Shelf@simd128", nil), types2.NewInterfaceType([]*types2.Func{read}, nil), nil)
			pkg.Scope().Insert(shelf.Obj())
			values := types2.NewFunc(pos, pkg, "Values@simd128", types2.NewSignatureType(nil, nil, nil, nil,
				types2.NewTuple(types2.NewVar(pos, pkg, "", types2.NewSlice(types2.Typ[types2.Int]))), false))
			pkg.Scope().Insert(values)
			var consts []facts.PathFact
			if structural {
				consts = []facts.PathFact{{Object: values.Name(), Path: "R0"}}
			}
			out, err := CloneWithOverlaysAndReadonly(pkg, nil, consts,
				[]facts.FnFact{{Object: fn.Name(), HasAssert: true, AssertTP: "1", AssertRTP: "0"}},
				[]facts.ReadonlyFact{{Object: shelf.Obj().Name(), Path: "MRead"}}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if (out != pkg) != structural {
				t.Fatalf("package replaced=%v, want %v", out != pkg, structural)
			}
			for _, object := range []string{fn.Name(), shelf.Obj().Name(), values.Name()} {
				if out.Scope().Lookup(object) == nil {
					t.Fatalf("clone lost tagged declaration %q", object)
				}
			}
			checkTaggedFactMasks(t, out.Scope().Lookup(fn.Name()).(*types2.Func), "1", "0")
			outRead := out.Scope().Lookup(shelf.Obj().Name()).Type().Underlying().(*types2.Interface).ExplicitMethod(0)
			if !outRead.LogosReadonly() {
				t.Fatal("clone lost tagged readonly type metadata")
			}
			if structural {
				sig := out.Scope().Lookup(values.Name()).Type().(*types2.Signature)
				if _, ok := sig.Results().At(0).Type().(*types2.ConstView); !ok {
					t.Fatalf("tagged declaration lost structural const overlay: %s", sig)
				}
				if _, _, known := fn.LogosAssertEffects(); known || read.LogosReadonly() {
					t.Fatal("structural clone published metadata into the original package")
				}
			} else if out.Scope().Lookup(fn.Name()) != fn || outRead != read {
				t.Fatal("declaration-only overlay replaced canonical objects")
			}
		})
	}
}

func TestFactScopeRejectsMalformedTaggedSymbols(t *testing.T) {
	for _, symbol := range []string{
		"Project@", "Project@1", "Project@simd128!", "Project@simd128@again", "@simd128", "for@simd128",
		"Project[go.shape.int]@simd128", "local·1@simd128", "Box.Method@simd128", "Project@simd128.Read",
	} {
		for _, operation := range []string{"functions", "readonly", "empty functions", "empty readonly", "declaration clone", "structural clone", "readonly symbol"} {
			t.Run(symbol+"/"+operation, func(t *testing.T) {
				pkg := types2.NewPackage("example.com/taggedfacts", "taggedfacts")
				pos := syntax.Pos{}
				public := taggedFactFunction(pkg, "Public")
				read := types2.NewFunc(pos, pkg, "Read", types2.NewSignatureType(nil, nil, nil, nil, nil, false))
				shelf := types2.NewNamed(types2.NewTypeName(pos, pkg, "Shelf", nil), types2.NewInterfaceType([]*types2.Func{read}, nil), nil)
				pkg.Scope().Insert(shelf.Obj())
				values := types2.NewFunc(pos, pkg, "Values", types2.NewSignatureType(nil, nil, nil, nil,
					types2.NewTuple(types2.NewVar(pos, pkg, "", types2.NewSlice(types2.Typ[types2.Int]))), false))
				pkg.Scope().Insert(values)
				pkg.Scope().InsertLazy(symbol, func() types2.Object {
					t.Fatalf("resolved unsupported tagged symbol %q during %s", symbol, operation)
					return nil
				})
				fns := []facts.FnFact{{Object: public.Name(), HasAssert: true, AssertTP: "1", AssertRTP: "0"}}
				readonly := []facts.ReadonlyFact{{Object: "Shelf", Path: "MRead"}}
				out := pkg
				var err error
				switch operation {
				case "functions":
					err = ApplyFuncFacts(pkg, append(fns, facts.FnFact{Object: symbol, HasAssert: true, AssertTP: "1", AssertRTP: "0"}))
				case "readonly":
					err = ApplyReadonlyFactsInPlace(pkg, readonly)
				case "empty functions":
					err = ApplyFuncFacts(pkg, nil)
				case "empty readonly":
					err = ApplyReadonlyFactsInPlace(pkg, nil)
				case "declaration clone":
					out, err = CloneWithOverlaysAndReadonly(pkg, nil, nil, fns, readonly, nil)
				case "structural clone":
					out, err = CloneWithOverlaysAndReadonly(pkg, nil, []facts.PathFact{{Object: "Values", Path: "R0"}}, fns, readonly, nil)
				case "readonly symbol":
					err = ApplyReadonlyFactsInPlace(pkg, append(readonly, facts.ReadonlyFact{Object: symbol, Path: "MRead"}))
				}
				if err != nil {
					t.Fatal(err)
				}
				for _, object := range []string{"Public", "Shelf", "Values"} {
					if out.Scope().Lookup(object) == nil {
						t.Fatalf("lost ordinary declaration %q while excluding %q", object, symbol)
					}
				}
				if operation == "functions" || operation == "declaration clone" || operation == "structural clone" {
					checkTaggedFactMasks(t, out.Scope().Lookup("Public").(*types2.Func), "1", "0")
				}
				if operation == "readonly" || operation == "readonly symbol" || operation == "declaration clone" || operation == "structural clone" {
					outRead := out.Scope().Lookup("Shelf").Type().Underlying().(*types2.Interface).ExplicitMethod(0)
					if !outRead.LogosReadonly() {
						t.Fatal("ordinary readonly metadata was lost")
					}
				}
				if operation == "structural clone" {
					if out == pkg {
						t.Fatal("structural overlay did not clone the package")
					}
					if _, ok := out.Scope().Lookup("Values").Type().(*types2.Signature).Results().At(0).Type().(*types2.ConstView); !ok {
						t.Fatal("ordinary structural const overlay was lost")
					}
				}
			})
		}
	}
}
