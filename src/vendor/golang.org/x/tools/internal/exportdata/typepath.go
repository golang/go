// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exportdata

// This file defines type paths, which shallow export data uses to
// refer to struct and interface literals declared in other packages.
//
// When doing a shallow import, the importer creates only the local
// package, and obtains the symbols of dependencies from the client.
// However, the types of the local package may contain struct or
// interface literals declared (perhaps deeply) within another package.
// For example:
//
//	package a
//	func F() chan *map[string]struct{ X int }
//
//	package b
//	import "a"
//	var B = a.F()
//
// Here the type of b.B contains a struct literal declared in package
// a. To preserve the identity of the struct type and its field
// objects, the exporter does not encode the literal, but instead
// records a path to it from the API of package a, and the importer
// follows that path within its symbols for package a.
//
// A path consists of the name of an exported package-level object,
// followed by a sequence of indices, each selecting a component of the
// current type (an element, field, parameter, method, and so on).
//
// Within the path's package, the traversal descends into the
// package's own named types, even unexported ones, whenever it
// reaches them; it does not descend into the named types of other
// packages, nor into instances. Export data includes the same set of
// declarations for a package, so every path found by the exporter is
// valid within any representation of the package, whether
// type-checked from source or imported from export data.
//
// The traversal does not descend into type parameters or their
// constraints: a literal reachable only through a constraint
// cannot appear in the types of another package.
//
// A literal created by instantiation (for example, the underlying type
// of b.B in "type B a.A[int]") cannot be reached by destructuring.
// Such literals are encoded in full; see the TODO in
// [exportWriter.externalLiteral].
//
// Unlike objectpath, this encoding is part of the export data
// format, and is versioned with it.

import (
	"go/types"
	"iter"
	"slices"
)

// A typePath denotes a type relative to its package, expressed as a
// sequence of type-destructuring operations applied to the type of a
// package-level object.
type typePath struct {
	root  string // name of an exported package-level object
	steps []int  // component indices, starting from the root object's type
}

// newTypeIndex returns the canonical paths of the literal struct and
// interface types reachable from the API of pkg.
//
// (Used during export.)
func newTypeIndex(pkg *types.Package) map[types.Type]typePath {
	var (
		paths = make(map[types.Type]typePath) // literal -> first path found
		seen  = make(map[*types.Named]bool)
		root  string // name of current root object
		visit func(t types.Type, stack []int)
	)
	visit = func(t types.Type, stack []int) {
		switch t := types.Unalias(t).(type) {
		case *types.Named:
			// Descend only into this package's own non-instantiated named types.
			// TODO(golang/go#61674): descending into instances too
			// would find instantiated literals reachable from them; see
			// the TODO in [exportWriter.externalLiteral].
			if t.Obj().Pkg() != pkg || t.TypeArgs().Len() > 0 || seen[t] {
				return
			}
			seen[t] = true

		case *types.Struct, *types.Interface:
			if _, ok := paths[t]; ok {
				return // already visited
			}
			paths[t] = typePath{root, slices.Clone(stack)}
		}

		// Traverse all child types.
		for i := 0; ; i++ {
			tch, ok := child(t, i)
			if !ok {
				break
			}
			visit(tch, append(stack, i))
		}
	}
	stack := make([]int, 0, 16)
	scope := pkg.Scope()
	for _, name := range scope.Names() { // (deterministic)
		obj := scope.Lookup(name)
		if obj.Exported() {
			root = obj.Name()
			visit(obj.Type(), stack)
		}
	}
	return paths
}

// literalMembers returns the fields of a struct literal, or the
// explicit methods of an interface literal.
func literalMembers(t types.Type) iter.Seq[types.Object] {
	return func(yield func(types.Object) bool) {
		switch t := t.(type) {
		case *types.Struct:
			t.Fields()(func(f *types.Var) bool { return yield(f) })
		case *types.Interface:
			t.ExplicitMethods()(func(m *types.Func) bool { return yield(m) })
		}
	}
}

// origin returns the origin of member obj (a field or method).
func origin(obj types.Object) types.Object {
	switch obj := obj.(type) {
	case *types.Var:
		return obj.Origin()
	case *types.Func:
		return obj.Origin()
	}
	return obj
}

// resolveTypePath returns the type denoted by path within pkg.
//
// (Used during import.)
func resolveTypePath(pkg *types.Package, path typePath) types.Type {
	obj := pkg.Scope().Lookup(path.root)
	if obj == nil {
		errorf("type path: package %s has no member %q", pkg.Path(), path.root)
	}
	t := obj.Type()
	for _, i := range path.steps {
		tch, ok := child(t, i)
		if !ok {
			errorf("type path %s.%s%v: no component %d of %s", pkg.Path(), path.root, path.steps, i, t)
		}
		t = tch
	}
	return types.Unalias(t)
}

// child returns the ith component type of t, or false if there is
// none. It is the sole definition of the meaning of a typePath step,
// used both by the exporter's traversal and by the importer's
// resolution. The components are, in order:
//
//   - pointer, slice, array, chan: the element type;
//   - map: the key and element types;
//   - signature: the parameter types, then the result types;
//   - struct: the field types;
//   - interface: the explicit method types;
//   - named: the underlying type, then the method types.
//
// Aliases are transparent.
func child(t types.Type, i int) (types.Type, bool) {
	switch t := types.Unalias(t).(type) {
	case *types.Pointer, *types.Slice, *types.Array, *types.Chan:
		if i == 0 {
			return t.(interface{ Elem() types.Type }).Elem(), true
		}
	case *types.Map:
		switch i {
		case 0:
			return t.Key(), true
		case 1:
			return t.Elem(), true
		}
	case *types.Signature:
		if n := t.Params().Len(); i < n {
			return t.Params().At(i).Type(), true
		} else if i-n < t.Results().Len() {
			return t.Results().At(i - n).Type(), true
		}
	case *types.Struct:
		if i < t.NumFields() {
			return t.Field(i).Type(), true
		}
	case *types.Interface:
		if i < t.NumExplicitMethods() {
			return t.ExplicitMethod(i).Type(), true
		}
	case *types.Named:
		if i == 0 {
			return t.Underlying(), true
		} else if i-1 < t.NumMethods() {
			return t.Method(i - 1).Type(), true
		}
	}
	return nil, false
}
