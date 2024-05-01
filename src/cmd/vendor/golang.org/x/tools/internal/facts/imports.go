// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package facts

import (
	"go/types"

	"golang.org/x/tools/internal/aliases"
)

// importMap computes the import map for a package by traversing the
// entire exported API each of its imports.
//
// This is a workaround for the fact that we cannot access the map used
// internally by the types.Importer returned by go/importer. The entries
// in this map are the packages and objects that may be relevant to the
// current analysis unit.
//
// Packages in the map that are only indirectly imported may be
// incomplete (!pkg.Complete()).
//
// This function scales very poorly with packages' transitive object
// references, which can be more than a million for each package near
// the top of a large project. (This was a significant contributor to
// #60621.)
// TODO(adonovan): opt: compute this information more efficiently
// by obtaining it from the internals of the gcexportdata decoder.
func importMap(imports []*types.Package) map[string]*types.Package {
	objects := make(map[types.Object]bool)
	typs := make(map[types.Type]bool) // Named and TypeParam
	packages := make(map[string]*types.Package)

	var addObj func(obj types.Object)
	var addType func(T types.Type)

	addObj = func(obj types.Object) {
		if !objects[obj] {
			objects[obj] = true
			addType(obj.Type())
			if pkg := obj.Pkg(); pkg != nil {
				packages[pkg.Path()] = pkg
			}
		}
	}

	addType = func(T types.Type) {
		switch T := T.(type) {
		case *aliases.Alias:
			addType(aliases.Unalias(T))
		case *types.Basic:
			// nop
		case *types.Named:
			// Remove infinite expansions of *types.Named by always looking at the origin.
			// Some named types with type parameters [that will not type check] have
			// infinite expansions:
			//     type N[T any] struct { F *N[N[T]] }
			// importMap() is called on such types when Analyzer.RunDespiteErrors is true.
			T = T.Origin()
			if !typs[T] {
				typs[T] = true
				addObj(T.Obj())
				addType(T.Underlying())
				for i := 0; i < T.NumMethods(); i++ {
					addObj(T.Method(i))
				}
				if tparams := T.TypeParams(); tparams != nil {
					for i := 0; i < tparams.Len(); i++ {
						addType(tparams.At(i))
					}
				}
				if targs := T.TypeArgs(); targs != nil {
					for i := 0; i < targs.Len(); i++ {
						addType(targs.At(i))
					}
				}
			}
		case *types.Pointer:
			addType(T.Elem())
		case *types.Slice:
			addType(T.Elem())
		case *types.Array:
			addType(T.Elem())
		case *types.Chan:
			addType(T.Elem())
		case *types.Map:
			addType(T.Key())
			addType(T.Elem())
		case *types.Signature:
			addType(T.Params())
			addType(T.Results())
			if tparams := T.TypeParams(); tparams != nil {
				for i := 0; i < tparams.Len(); i++ {
					addType(tparams.At(i))
				}
			}
		case *types.Struct:
			for i := 0; i < T.NumFields(); i++ {
				addObj(T.Field(i))
			}
		case *types.Tuple:
			for i := 0; i < T.Len(); i++ {
				addObj(T.At(i))
			}
		case *types.Interface:
			for i := 0; i < T.NumMethods(); i++ {
				addObj(T.Method(i))
			}
			for i := 0; i < T.NumEmbeddeds(); i++ {
				addType(T.EmbeddedType(i)) // walk Embedded for implicits
			}
		case *types.Union:
			for i := 0; i < T.Len(); i++ {
				addType(T.Term(i).Type())
			}
		case *types.TypeParam:
			if !typs[T] {
				typs[T] = true
				addObj(T.Obj())
				addType(T.Constraint())
			}
		}
	}

	for _, imp := range imports {
		packages[imp.Path()] = imp

		scope := imp.Scope()
		for _, name := range scope.Names() {
			addObj(scope.Lookup(name))
		}
	}

	return packages
}
