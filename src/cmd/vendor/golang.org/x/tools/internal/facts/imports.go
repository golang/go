// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package facts

import (
	"go/types"

	"golang.org/x/tools/internal/aliases"
	"golang.org/x/tools/internal/typesinternal"
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
		case *types.Basic:
			// nop
		case typesinternal.NamedOrAlias: // *types.{Named,Alias}
			// Add the type arguments if this is an instance.
			if targs := T.TypeArgs(); targs.Len() > 0 {
				for t := range targs.Types() {
					addType(t)
				}
			}

			// Remove infinite expansions of *types.Named by always looking at the origin.
			// Some named types with type parameters [that will not type check] have
			// infinite expansions:
			//     type N[T any] struct { F *N[N[T]] }
			// importMap() is called on such types when Analyzer.RunDespiteErrors is true.
			T = typesinternal.Origin(T)
			if !typs[T] {
				typs[T] = true

				// common aspects
				addObj(T.Obj())
				if tparams := T.TypeParams(); tparams.Len() > 0 {
					for tparam := range tparams.TypeParams() {
						addType(tparam)
					}
				}

				// variant aspects
				switch T := T.(type) {
				case *types.Alias:
					addType(aliases.Rhs(T))
				case *types.Named:
					addType(T.Underlying())
					for method := range T.Methods() {
						addObj(method)
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
				for tparam := range tparams.TypeParams() {
					addType(tparam)
				}
			}
		case *types.Struct:
			for field := range T.Fields() {
				addObj(field)
			}
		case *types.Tuple:
			for v := range T.Variables() {
				addObj(v)
			}
		case *types.Interface:
			for method := range T.Methods() {
				addObj(method)
			}
			for etyp := range T.EmbeddedTypes() {
				addType(etyp) // walk Embedded for implicits
			}
		case *types.Union:
			for term := range T.Terms() {
				addType(term.Type())
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
