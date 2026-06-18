// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package midway

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// Analyzer holds the state for SIMD dependency analysis
type Analyzer struct {
	pkg                *types2.Package
	info               *types2.Info
	isDependentObj     map[types2.Object]bool // does an Object depend on a simd type in some way?
	isDependentMethod  map[types2.Object]bool // is this dependent Object also a method? (methods are not renamed, their types are)
	hasDependentMethod map[types2.Type]bool   // is this a type that has a dependent method?
	visited            map[types2.Type]bool   // if in map, type has been visited, and value is whether type is dependent
	inSimd             bool                   // true if the current package is the simd package (which is a special case)
}

func NewAnalyzer(pkg *types2.Package, info *types2.Info) *Analyzer {
	return &Analyzer{
		pkg:                pkg,
		info:               info,
		isDependentObj:     make(map[types2.Object]bool),
		isDependentMethod:  make(map[types2.Object]bool),
		hasDependentMethod: make(map[types2.Type]bool),
		visited:            make(map[types2.Type]bool),
		inSimd:             pkg.Path() == simdPkg,
	}
}

// Analyze builds the set of SIMD-dependent objects
func (a *Analyzer) Analyze(files []*syntax.File) bool {
	// Phase 1: Seed dependence from types and signatures
	hdmsize := len(a.hasDependentMethod)

	for {
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
		if hdmsize == len(a.hasDependentMethod) {
			break
		}
		if base.Debug.Simd > 0 {
			base.Warn("hasDependentMethod increased from %d to %d", hdmsize, len(a.hasDependentMethod))
		}
		hdmsize = len(a.hasDependentMethod)
		clear(a.visited)
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
					if obj == nil || a.isDependentObj[obj] {
						continue
					}

					if a.hasBodyDependency(fn) {
						a.isDependentObj[obj] = true
						changed = true
					}
				}
			}
		}
	}

	return len(a.isDependentObj) > 0
}

func (a *Analyzer) hasBodyDependency(fn *syntax.FuncDecl) bool {
	if fn.Body == nil {
		return false
	}
	// Walk the body and check identifiers
	// This will also note any variable references that are dependent.
	found := false
	syntax.Inspect(fn.Body, func(n syntax.Node) bool {
		if id, ok := n.(*syntax.Name); ok {
			obj := a.info.Uses[id]
			if obj == nil {
				obj = a.info.Defs[id]
			}
			if obj != nil {
				if _, isFunc := obj.(*types2.Func); !isFunc {
					if a.isDependentObj[obj] {
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
					// Whatever this is, it makes the outer object dependent.
					// If this is a package variable with dependent type, mark the
					// variable as dependent, so that references to it become dependent.
					if obj, ok := obj.(*types2.Var); ok && obj.Kind() == types2.PackageVar {
						// everything else is nested within a dependent function/struct/scope
						// and does not need its own renaming
						a.isDependentObj[obj] = true
					}
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
	if a.isDependentObj[obj] {
		return true
	}

	isDep := false
	isDepMeth := false
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
				t := rcv.Type()
				if !a.isDependentType(t) {
					a.markHasMethod(t)
				}
				isDepMeth = true
			}
		}
	}

	// Also check if obj name is "simd.Type" (base case)
	if isBaseSimdTypeObj(obj) {
		isDep = true
	}

	if isDep {
		if base.Debug.Simd > 0 {
			base.Warn("%s: %v is simd-dependent", obj.Pos().String(), obj)
		}
		a.isDependentObj[obj] = true
	}
	if isDepMeth {
		if base.Debug.Simd > 0 {
			base.Warn("%s: %v is simd-dependent method", obj.Pos().String(), obj)
		}
		a.isDependentMethod[obj] = true
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
	if a.hasDependentMethod[t] {
		a.visited[t] = true
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

// This attempts to mark types that are not otherwise dependent
// as being dependent, if they have a method with a dependent
// signature.
func (a *Analyzer) markHasMethod(t types2.Type) {
	if t == nil {
		return
	}
	if a.hasDependentMethod[t] {
		return
	}

	a.hasDependentMethod[t] = true

	switch t := t.(type) {
	case *types2.Pointer:
		a.markHasMethod(t.Elem())
	case *types2.Alias:
		a.markHasMethod(t.Rhs())
	}
	return
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
