// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package midway

import (
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// Analyzer holds the state for SIMD dependency analysis
type Analyzer struct {
	pkg          *types2.Package
	info         *types2.Info
	dependentObj map[types2.Object]bool
	visited      map[types2.Type]bool
	inSimd       bool
}

func NewAnalyzer(pkg *types2.Package, info *types2.Info) *Analyzer {
	return &Analyzer{
		pkg:          pkg,
		info:         info,
		dependentObj: make(map[types2.Object]bool),
		visited:      make(map[types2.Type]bool),
		inSimd:       pkg.Path() == simdPkg,
	}
}

// Analyze builds the set of SIMD-dependent objects
func (a *Analyzer) Analyze(files []*syntax.File) bool {
	// Phase 1: Seed dependence from types and signatures
	for _, obj := range a.info.Defs {
		if obj != nil {
			a.markIfDependent(obj)
		}
	}
	for _, obj := range a.info.Uses {
		if obj != nil {
			a.markIfDependent(obj)
		}
	}

	// Phase 2: Transitive closure via function bodies
	changed := true
	for changed {
		changed = false
		for _, file := range files {
			for _, decl := range file.DeclList {
				if fn, ok := decl.(*syntax.FuncDecl); ok {
					if fn.Name == nil {
						continue
					}
					obj := a.info.Defs[fn.Name]
					if obj == nil || a.dependentObj[obj] {
						continue
					}

					if a.hasBodyDependency(fn) {
						a.dependentObj[obj] = true
						changed = true
					}
				}
			}
		}
	}

	return len(a.dependentObj) > 0
}

func (a *Analyzer) hasBodyDependency(fn *syntax.FuncDecl) bool {
	if fn.Body == nil {
		return false
	}
	// Walk the body and check identifiers
	found := false
	syntax.Inspect(fn.Body, func(n syntax.Node) bool {
		if found {
			return false
		}
		if id, ok := n.(*syntax.Name); ok {
			obj := a.info.Uses[id]
			if obj == nil {
				obj = a.info.Defs[id]
			}
			if obj != nil {
				if _, isFunc := obj.(*types2.Func); !isFunc {
					if a.dependentObj[obj] {
						found = true
						return false
					}
				} else {
					sig := obj.Type().(*types2.Signature)
					if a.HasDependentSignature(sig) {
						found = true
						return false
					}
				}
				if a.isDependentType(obj.Type()) {
					found = true
					return false
				}
				if isBaseSimdTypeObj(obj) {
					found = true
					return false
				}
			}
		}
		return true
	})
	return found
}

func (a *Analyzer) markIfDependent(obj types2.Object) bool {
	if a.dependentObj[obj] {
		return true
	}

	isDep := false
	switch obj := obj.(type) {
	case *types2.Var:
		if obj.Pkg() == a.pkg && obj.Parent() == a.pkg.Scope() {
			isDep = a.isDependentType(obj.Type())
		}
	case *types2.TypeName:
		isDep = a.isDependentType(obj.Type())
	case *types2.Func:
		sig := obj.Type().(*types2.Signature)
		if a.HasDependentSignature(sig) {
			// NOT dependent if it is a method of one of the base SIMD types.
			// TODO: what about aliases of base SIMD types?
			if rcv := sig.Recv(); rcv == nil {
				isDep = true
			} else if named, ok := rcv.Type().(*types2.Named); !ok || !isBaseSimdType(named) {
				isDep = true
			}
		}
	}

	// Also check if obj name is "simd.Type" (base case)
	if isBaseSimdTypeObj(obj) {
		isDep = true
	}

	if isDep {
		a.dependentObj[obj] = true
	}
	return isDep
}

func (a *Analyzer) isDependentType(t types2.Type) bool {
	return a.checkTypeRecursive(t)
}

func (a *Analyzer) checkTypeRecursive(t types2.Type) bool {
	if t == nil {
		return false
	}
	if b, ok := a.visited[t]; ok {
		return b // Break cycles
	}
	a.visited[t] = false

	memo := func(b bool) bool {
		a.visited[t] = b
		return b
	}

	// Unwrap aliases
	if named, ok := t.(*types2.Named); ok {
		if isBaseSimdType(named) {
			return memo(true)
		}
		if a.checkTypeRecursive(named.Underlying()) {
			return memo(true)
		}
	}

	switch t := t.(type) {
	case *types2.Basic:
		return false
	case *types2.Pointer:
		return memo(a.checkTypeRecursive(t.Elem()))
	case *types2.Slice:
		return memo(a.checkTypeRecursive(t.Elem()))
	case *types2.Array:
		return memo(a.checkTypeRecursive(t.Elem()))
	case *types2.Map:
		return memo(a.checkTypeRecursive(t.Key()) ||
			a.checkTypeRecursive(t.Elem()))
	case *types2.Chan:
		return memo(a.checkTypeRecursive(t.Elem()))
	case *types2.Struct:
		for i := 0; i < t.NumFields(); i++ {
			if a.checkTypeRecursive(t.Field(i).Type()) {
				return memo(true)
			}
		}
	case *types2.Signature:
		return memo(a.HasDependentSignature(t))
	case *types2.Tuple:
		for i := 0; i < t.Len(); i++ {
			if a.checkTypeRecursive(t.At(i).Type()) {
				return memo(true)
			}
		}
	case *types2.Alias:
		return memo(a.checkTypeRecursive(types2.Unalias(t)))
	}
	return false
}

func isBaseSimdType(t *types2.Named) bool {
	return isBaseSimdTypeObj(t.Obj())
}

func isBaseSimdTypeObj(obj types2.Object) bool {
	if obj == nil || obj.Pkg() == nil {
		return false
	}
	if obj.Pkg().Path() != simdPkg {
		return false
	}
	return isSimdTypeName(obj.Name())
}

func (a *Analyzer) HasDependentSignature(sig *types2.Signature) bool {
	// TODO what about type parameters?  Need to invent a test that provokes that case.
	return a.isDependentType(sig.Params()) ||
		a.isDependentType(sig.Results()) ||
		(sig.Recv() != nil && a.isDependentType(sig.Recv().Type()))
}
