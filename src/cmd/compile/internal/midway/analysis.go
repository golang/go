// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package midway

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"iter"
	"math"
)

// Analyzer holds the state for SIMD dependency analysis.
type Analyzer struct {
	pkg               *types2.Package
	info              *types2.Info
	isDependentObj    map[types2.Object]bool // does an Object depend on a simd type in some way?
	isDependentMethod map[types2.Object]bool // is this dependent Object also a method? (methods are not renamed, their types are)
	visited           map[types2.Type]bool   // finalized memoization of isDependentType
	inSimd            bool                   // true if the current package is the simd package (which is a special case)
}

func NewAnalyzer(pkg *types2.Package, info *types2.Info) *Analyzer {
	return &Analyzer{
		pkg:               pkg,
		info:              info,
		isDependentObj:    make(map[types2.Object]bool),
		isDependentMethod: make(map[types2.Object]bool),
		visited:           make(map[types2.Type]bool),
		inSimd:            pkg.Path() == SimdPkg,
	}
}

// Analyze builds the set of SIMD-dependent objects.
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
					if fn.Name == nil || fn.Name.Value == "_" {
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
				// A function that is dependent only through its body gets a
				// dispatcher, so calling it doesn't make the caller dependent.
				// A function with a dependent signature is caught by the
				// isDependentType check below.
				if _, isFunc := obj.(*types2.Func); !isFunc {
					if a.isDependentObj[obj] {
						found = true
						return false
					}
				}
				if a.isDependentType(obj.Type()) {
					// Whatever this is, it makes the outer object dependent.
					// If this is a package variable with dependent type, mark the
					// variable as dependent, so that references to it become dependent.
					if obj, ok := obj.(*types2.Var); ok && isPkgScope(obj) {
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

// isPkgScope reports whether obj is a non-blank package-level declaration.
// Blank declarations have a nil parent; predeclared objects have a nil Pkg.
func isPkgScope(obj types2.Object) bool {
	return obj.Pkg() != nil && obj.Parent() == obj.Pkg().Scope()
}

func (a *Analyzer) markIfDependent(obj types2.Object) bool {
	if a.isDependentObj[obj] {
		return true
	}

	isDep := false
	isDepMeth := false
	switch obj := obj.(type) {
	case *types2.Var:
		if isPkgScope(obj) {
			isDep = a.isDependentType(obj.Type())
		}
	case *types2.TypeName:
		// Only non-blank package-level type declarations are specialized and
		// suffixed. Type parameters live in their declaration's type-param
		// scope, local types live in a block scope, and blank "_" declarations
		// (both type parameters and "type _ = ...") have a nil parent.
		if isPkgScope(obj) {
			isDep = a.isDependentType(obj.Type())
		}
	case *types2.Func:
		// Blank functions and methods are never callable and are skipped by
		// noder, so they must not be specialized or suffixed (which would
		// cause multiple "_" declarations to collide as "_@simd*").
		if obj.Name() == "_" {
			break
		}
		sig := obj.Signature()
		if a.HasDependentSignature(sig) {
			// NOT dependent if it is a method of one of the base SIMD types.
			// TODO: what about aliases of base SIMD types?
			if rcv := sig.Recv(); rcv == nil {
				isDep = true
			} else if !isBaseSimdType(rcv.Type()) {
				isDep = true
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

// sccs yields the strongly connected components reachable from root in
// reverse topological order (dependencies before dependents), using Tarjan's
// algorithm. The yielded slice aliases internal stack storage and is valid
// only until the next iteration.
func sccs[T comparable](root T, edges func(T) iter.Seq[T]) iter.Seq[[]T] {
	return func(yield func([]T) bool) {
		idx := make(map[T]int)
		var stack []T
		nextIdx := 1
		const done = math.MaxInt

		var dfs func(v T) (low int, cont bool)
		dfs = func(v T) (int, bool) {
			i := nextIdx
			nextIdx++
			idx[v] = i
			low := i
			top := len(stack)
			stack = append(stack, v)

			for w := range edges(v) {
				if wi := idx[w]; wi == 0 {
					wLow, ok := dfs(w)
					if !ok {
						return 0, false
					}
					low = min(low, wLow)
				} else if wi < done {
					low = min(low, wi)
				}
			}

			if low == i {
				scc := stack[top:]
				for _, x := range scc {
					idx[x] = done
				}
				cont := yield(scc)
				stack = stack[:top]
				if !cont {
					return 0, false
				}
			}
			return low, true
		}

		dfs(root)
	}
}

// visitType yields the types directly referenced by t.
// For instantiated *types2.Named and *types2.Alias types, it yields only the
// generic origin and type arguments, avoiding lazy expansion of *types2.Named
// instances and redundant traversal of substituted alias RHSes.
func visitType(t types2.Type) iter.Seq[types2.Type] {
	return func(yield func(types2.Type) bool) {
		switch t := t.(type) {
		case *types2.Basic:
		case *types2.Named:
			if orig := t.Origin(); orig != t {
				if !yield(orig) {
					return
				}
				for i := range t.TypeArgs().Len() {
					if !yield(t.TypeArgs().At(i)) {
						return
					}
				}
				return
			}
			if !yield(t.Underlying()) {
				return
			}
			for i := range t.TypeParams().Len() {
				if !yield(t.TypeParams().At(i)) {
					return
				}
			}
			// Inspect methods so types whose SIMD dependence appears only in
			// their method set (not their fields) are still detected.
			for i := range t.NumMethods() {
				if !yield(t.Method(i).Signature()) {
					return
				}
			}
		case *types2.Alias:
			if orig := t.Origin(); orig != t {
				if !yield(orig) {
					return
				}
				for i := range t.TypeArgs().Len() {
					if !yield(t.TypeArgs().At(i)) {
						return
					}
				}
				return
			}
			if !yield(t.Rhs()) {
				return
			}
			for i := range t.TypeParams().Len() {
				if !yield(t.TypeParams().At(i)) {
					return
				}
			}
		case *types2.TypeParam:
			yield(t.Constraint())
		case *types2.Pointer:
			yield(t.Elem())
		case *types2.Slice:
			yield(t.Elem())
		case *types2.Array:
			yield(t.Elem())
		case *types2.Map:
			if !yield(t.Key()) {
				return
			}
			yield(t.Elem())
		case *types2.Chan:
			yield(t.Elem())
		case *types2.Struct:
			for i := range t.NumFields() {
				if !yield(t.Field(i).Type()) {
					return
				}
			}
		case *types2.Signature:
			for i := range t.TypeParams().Len() {
				if !yield(t.TypeParams().At(i)) {
					return
				}
			}
			for i := range t.RecvTypeParams().Len() {
				if !yield(t.RecvTypeParams().At(i)) {
					return
				}
			}
			for i := range t.Params().Len() {
				if !yield(t.Params().At(i).Type()) {
					return
				}
			}
			for i := range t.Results().Len() {
				if !yield(t.Results().At(i).Type()) {
					return
				}
			}
			if recv := t.Recv(); recv != nil {
				yield(recv.Type())
			}
		case *types2.Tuple:
			for i := range t.Len() {
				if !yield(t.At(i).Type()) {
					return
				}
			}
		case *types2.Interface:
			for i := range t.NumExplicitMethods() {
				if !yield(t.ExplicitMethod(i).Signature()) {
					return
				}
			}
			for i := range t.NumEmbeddeds() {
				if !yield(t.EmbeddedType(i)) {
					return
				}
			}
		case *types2.Union:
			for i := range t.Len() {
				if !yield(t.Term(i).Type()) {
					return
				}
			}
		default:
			base.Fatalf("unexpected type %T", t)
		}
	}
}

// isDependentType reports whether root transitively depends on a portable SIMD
// type through its definition, generic parameters/arguments, constraints, or
// methods. Results are finalized per strongly connected component and memoized
// in a.visited so cyclic types are resolved accurately.
func (a *Analyzer) isDependentType(root types2.Type) bool {
	if root == nil {
		return false
	}
	if b, ok := a.visited[root]; ok {
		return b
	}
	if _, ok := root.(*types2.Basic); ok {
		a.visited[root] = false
		return false
	}
	if isBaseSimdType(root) {
		a.visited[root] = true
		return true
	}

	// Prune edges to types whose dependence is already final in a.visited:
	// either memoized by an earlier call, or in an SCC already yielded (and
	// finalized below) during this walk. sccs has already marked the latter
	// done and would ignore them anyway, so pruning only avoids re-walking
	// resolved type graphs. The dependence check below still consults
	// a.visited for every edge, including pruned ones.
	unvisitedEdges := func(t types2.Type) iter.Seq[types2.Type] {
		return func(yield func(types2.Type) bool) {
			if isBaseSimdType(t) {
				return
			}
			for u := range visitType(t) {
				if _, ok := a.visited[u]; ok {
					continue
				}
				if !yield(u) {
					return
				}
			}
		}
	}

	for scc := range sccs(root, unvisitedEdges) {
		// The whole SCC is either dependent or not. Check if any type in the
		// SCC is a SIMD type itself, or directly references a SIMD-dependent
		// type.
		dep := false
	check:
		for _, t := range scc {
			if isBaseSimdType(t) {
				dep = true
				break
			}
			for u := range visitType(t) {
				if a.visited[u] {
					dep = true
					break check
				}
			}
		}
		// Finalize the whole SCC.
		for _, t := range scc {
			a.visited[t] = dep
		}
	}
	return a.visited[root]
}

// isBaseSimdType reports whether t is a base portable SIMD type (including the
// internal simd._simd marker type embedded in all portable SIMD types, which
// allows detecting defined types such as "type MySimd simd.Int8s").
func isBaseSimdType(t types2.Type) bool {
	named, ok := t.(*types2.Named)
	if !ok {
		return false
	}
	obj := named.Obj()
	if obj == nil || obj.Pkg() == nil || obj.Pkg().Path() != SimdPkg {
		return false
	}
	return obj.Name() == "_simd" || isSimdTypeName(obj.Name())
}

func isBaseSimdTypeObj(obj types2.Object) bool {
	if obj == nil || obj.Pkg() == nil {
		return false
	}
	if obj.Pkg().Path() != SimdPkg {
		return false
	}
	return isSimdTypeName(obj.Name())
}

// HasDependentSignature reports whether sig depends on a portable SIMD type.
func (a *Analyzer) HasDependentSignature(sig *types2.Signature) bool {
	if sig == nil {
		return false
	}
	return a.isDependentType(sig)
}
