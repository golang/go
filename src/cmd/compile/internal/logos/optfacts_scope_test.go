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

// Imported scopes include lazy linker symbols. Resolving a generic local type
// through the public importer is invalid, even when it appears in Scope.Names.
func TestFactScopeLeavesCompilerSymbolsLazy(t *testing.T) {
	for _, symbol := range []string{"local·1", "hidden·23", "Box.Method", "Box[go.shape.int]"} {
		for _, operation := range []string{"functions", "readonly", "empty functions", "empty readonly", "declaration clone", "structural clone", "readonly symbol"} {
			t.Run(symbol+"/"+operation, func(t *testing.T) {
				pkg := types2.NewPackage("example.com/factscope", "factscope")
				pos := syntax.Pos{}
				fnFacts := []facts.FnFact{}
				for _, name := range []string{"Public", "private", "Ω", "_value2"} {
					fn := types2.NewFunc(pos, pkg, name, types2.NewSignatureType(nil, nil, nil, nil, nil, false))
					pkg.Scope().Insert(fn)
					fnFacts = append(fnFacts, facts.FnFact{Object: name, Readonly: true})
				}
				read := types2.NewFunc(pos, pkg, "Read", types2.NewSignatureType(nil, nil, nil, nil, nil, false))
				iface := types2.NewInterfaceType([]*types2.Func{read}, nil)
				shelf := types2.NewNamed(types2.NewTypeName(pos, pkg, "Shelf", nil), iface, nil)
				pkg.Scope().Insert(shelf.Obj())
				values := types2.NewFunc(pos, pkg, "Values", types2.NewSignatureType(nil, nil, nil, nil,
					types2.NewTuple(types2.NewVar(pos, pkg, "", types2.NewSlice(types2.Typ[types2.Int]))), false))
				pkg.Scope().Insert(values)
				pkg.Scope().InsertLazy(symbol, func() types2.Object {
					t.Fatalf("resolved compiler-only symbol %q during %s", symbol, operation)
					return nil
				})
				readonly := []facts.ReadonlyFact{{Object: "Shelf", Path: "MRead"}}
				out := pkg
				var err error
				switch operation {
				case "functions":
					err = ApplyFuncFacts(pkg, append(fnFacts, facts.FnFact{Object: symbol, Readonly: true}))
				case "readonly":
					err = ApplyReadonlyFactsInPlace(pkg, readonly)
				case "empty functions":
					err = ApplyFuncFacts(pkg, nil)
				case "empty readonly":
					err = ApplyReadonlyFactsInPlace(pkg, nil)
				case "declaration clone":
					out, err = CloneWithOverlaysAndReadonly(pkg, nil, nil, fnFacts, readonly, nil)
				case "structural clone":
					out, err = CloneWithOverlaysAndReadonly(pkg, nil,
						[]facts.PathFact{{Object: "Values", Path: "R0"}}, fnFacts, readonly, nil)
				case "readonly symbol":
					err = ApplyReadonlyFactsInPlace(pkg, append(readonly, facts.ReadonlyFact{Object: symbol, Path: "MRead"}))
				}
				if err != nil {
					t.Fatal(err)
				}
				for _, name := range []string{"Public", "private", "Ω", "_value2"} {
					fn, ok := out.Scope().Lookup(name).(*types2.Func)
					if !ok {
						t.Fatalf("lost source declaration %q", name)
					}
					if operation == "functions" || operation == "declaration clone" || operation == "structural clone" {
						if !fn.LogosReadonly() {
							t.Errorf("lost readonly metadata on source declaration %q", name)
						}
					}
				}
				if operation == "readonly" || operation == "readonly symbol" || operation == "declaration clone" || operation == "structural clone" {
					got := out.Scope().Lookup("Shelf").Type().Underlying().(*types2.Interface).ExplicitMethod(0)
					if !got.LogosReadonly() {
						t.Error("lost reachable interface method metadata")
					}
				}
				if operation == "structural clone" {
					if out == pkg {
						t.Fatal("structural overlay did not clone the package")
					}
					sig := out.Scope().Lookup("Values").Type().(*types2.Signature)
					if _, ok := sig.Results().At(0).Type().(*types2.ConstView); !ok {
						t.Fatalf("lost const overlay: %s", sig)
					}
				}
			})
		}
	}
}
