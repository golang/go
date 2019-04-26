// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package objectpath defines a naming scheme for types.Objects
// (that is, named entities in Go programs) relative to their enclosing
// package.
//
// Type-checker objects are canonical, so they are usually identified by
// their address in memory (a pointer), but a pointer has meaning only
// within one address space. By contrast, objectpath names allow the
// identity of an object to be sent from one program to another,
// establishing a correspondence between types.Object variables that are
// distinct but logically equivalent.
//
// A single object may have multiple paths. In this example,
//     type A struct{ X int }
//     type B A
// the field X has two paths due to its membership of both A and B.
// The For(obj) function always returns one of these paths, arbitrarily
// but consistently.
package objectpath

import (
	"fmt"
	"strconv"
	"strings"

	"go/types"
)

// A Path is an opaque name that identifies a types.Object
// relative to its package. Conceptually, the name consists of a
// sequence of destructuring operations applied to the package scope
// to obtain the original object.
// The name does not include the package itself.
type Path string

// Encoding
//
// An object path is a textual and (with training) human-readable encoding
// of a sequence of destructuring operators, starting from a types.Package.
// The sequences represent a path through the package/object/type graph.
// We classify these operators by their type:
//
//   PO package->object	Package.Scope.Lookup
//   OT  object->type 	Object.Type
//   TT    type->type 	Type.{Elem,Key,Params,Results,Underlying} [EKPRU]
//   TO   type->object	Type.{At,Field,Method,Obj} [AFMO]
//
// All valid paths start with a package and end at an object
// and thus may be defined by the regular language:
//
//   objectpath = PO (OT TT* TO)*
//
// The concrete encoding follows directly:
// - The only PO operator is Package.Scope.Lookup, which requires an identifier.
// - The only OT operator is Object.Type,
//   which we encode as '.' because dot cannot appear in an identifier.
// - The TT operators are encoded as [EKPRU].
// - The OT operators are encoded as [AFMO];
//   three of these (At,Field,Method) require an integer operand,
//   which is encoded as a string of decimal digits.
//   These indices are stable across different representations
//   of the same package, even source and export data.
//
// In the example below,
//
//	package p
//
//	type T interface {
//		f() (a string, b struct{ X int })
//	}
//
// field X has the path "T.UM0.RA1.F0",
// representing the following sequence of operations:
//
//    p.Lookup("T")					T
//    .Type().Underlying().Method(0).			f
//    .Type().Results().At(1)				b
//    .Type().Field(0)					X
//
// The encoding is not maximally compact---every R or P is
// followed by an A, for example---but this simplifies the
// encoder and decoder.
//
const (
	// object->type operators
	opType = '.' // .Type()		  (Object)

	// type->type operators
	opElem       = 'E' // .Elem()		(Pointer, Slice, Array, Chan, Map)
	opKey        = 'K' // .Key()		(Map)
	opParams     = 'P' // .Params()		(Signature)
	opResults    = 'R' // .Results()	(Signature)
	opUnderlying = 'U' // .Underlying()	(Named)

	// type->object operators
	opAt     = 'A' // .At(i)		(Tuple)
	opField  = 'F' // .Field(i)		(Struct)
	opMethod = 'M' // .Method(i)		(Named or Interface; not Struct: "promoted" names are ignored)
	opObj    = 'O' // .Obj()		(Named)
)

// The For function returns the path to an object relative to its package,
// or an error if the object is not accessible from the package's Scope.
//
// The For function guarantees to return a path only for the following objects:
// - package-level types
// - exported package-level non-types
// - methods
// - parameter and result variables
// - struct fields
// These objects are sufficient to define the API of their package.
// The objects described by a package's export data are drawn from this set.
//
// For does not return a path for predeclared names, imported package
// names, local names, and unexported package-level names (except
// types).
//
// Example: given this definition,
//
//	package p
//
//	type T interface {
//		f() (a string, b struct{ X int })
//	}
//
// For(X) would return a path that denotes the following sequence of operations:
//
//    p.Scope().Lookup("T")				(TypeName T)
//    .Type().Underlying().Method(0).			(method Func f)
//    .Type().Results().At(1)				(field Var b)
//    .Type().Field(0)					(field Var X)
//
// where p is the package (*types.Package) to which X belongs.
func For(obj types.Object) (Path, error) {
	pkg := obj.Pkg()

	// This table lists the cases of interest.
	//
	// Object				Action
	// ------                               ------
	// nil					reject
	// builtin				reject
	// pkgname				reject
	// label				reject
	// var
	//    package-level			accept
	//    func param/result			accept
	//    local				reject
	//    struct field			accept
	// const
	//    package-level			accept
	//    local				reject
	// func
	//    package-level			accept
	//    init functions			reject
	//    concrete method			accept
	//    interface method			accept
	// type
	//    package-level			accept
	//    local				reject
	//
	// The only accessible package-level objects are members of pkg itself.
	//
	// The cases are handled in four steps:
	//
	// 1. reject nil and builtin
	// 2. accept package-level objects
	// 3. reject obviously invalid objects
	// 4. search the API for the path to the param/result/field/method.

	// 1. reference to nil or builtin?
	if pkg == nil {
		return "", fmt.Errorf("predeclared %s has no path", obj)
	}
	scope := pkg.Scope()

	// 2. package-level object?
	if scope.Lookup(obj.Name()) == obj {
		// Only exported objects (and non-exported types) have a path.
		// Non-exported types may be referenced by other objects.
		if _, ok := obj.(*types.TypeName); !ok && !obj.Exported() {
			return "", fmt.Errorf("no path for non-exported %v", obj)
		}
		return Path(obj.Name()), nil
	}

	// 3. Not a package-level object.
	//    Reject obviously non-viable cases.
	switch obj := obj.(type) {
	case *types.Const, // Only package-level constants have a path.
		*types.TypeName, // Only package-level types have a path.
		*types.Label,    // Labels are function-local.
		*types.PkgName:  // PkgNames are file-local.
		return "", fmt.Errorf("no path for %v", obj)

	case *types.Var:
		// Could be:
		// - a field (obj.IsField())
		// - a func parameter or result
		// - a local var.
		// Sadly there is no way to distinguish
		// a param/result from a local
		// so we must proceed to the find.

	case *types.Func:
		// A func, if not package-level, must be a method.
		if recv := obj.Type().(*types.Signature).Recv(); recv == nil {
			return "", fmt.Errorf("func is not a method: %v", obj)
		}
		// TODO(adonovan): opt: if the method is concrete,
		// do a specialized version of the rest of this function so
		// that it's O(1) not O(|scope|).  Basically 'find' is needed
		// only for struct fields and interface methods.

	default:
		panic(obj)
	}

	// 4. Search the API for the path to the var (field/param/result) or method.

	// First inspect package-level named types.
	// In the presence of path aliases, these give
	// the best paths because non-types may
	// refer to types, but not the reverse.
	empty := make([]byte, 0, 48) // initial space
	for _, name := range scope.Names() {
		o := scope.Lookup(name)
		tname, ok := o.(*types.TypeName)
		if !ok {
			continue // handle non-types in second pass
		}

		path := append(empty, name...)
		path = append(path, opType)

		T := o.Type()

		if tname.IsAlias() {
			// type alias
			if r := find(obj, T, path); r != nil {
				return Path(r), nil
			}
		} else {
			// defined (named) type
			if r := find(obj, T.Underlying(), append(path, opUnderlying)); r != nil {
				return Path(r), nil
			}
		}
	}

	// Then inspect everything else:
	// non-types, and declared methods of defined types.
	for _, name := range scope.Names() {
		o := scope.Lookup(name)
		path := append(empty, name...)
		if _, ok := o.(*types.TypeName); !ok {
			if o.Exported() {
				// exported non-type (const, var, func)
				if r := find(obj, o.Type(), append(path, opType)); r != nil {
					return Path(r), nil
				}
			}
			continue
		}

		// Inspect declared methods of defined types.
		if T, ok := o.Type().(*types.Named); ok {
			path = append(path, opType)
			for i := 0; i < T.NumMethods(); i++ {
				m := T.Method(i)
				path2 := appendOpArg(path, opMethod, i)
				if m == obj {
					return Path(path2), nil // found declared method
				}
				if r := find(obj, m.Type(), append(path2, opType)); r != nil {
					return Path(r), nil
				}
			}
		}
	}

	return "", fmt.Errorf("can't find path for %v in %s", obj, pkg.Path())
}

func appendOpArg(path []byte, op byte, arg int) []byte {
	path = append(path, op)
	path = strconv.AppendInt(path, int64(arg), 10)
	return path
}

// find finds obj within type T, returning the path to it, or nil if not found.
func find(obj types.Object, T types.Type, path []byte) []byte {
	switch T := T.(type) {
	case *types.Basic, *types.Named:
		// Named types belonging to pkg were handled already,
		// so T must belong to another package. No path.
		return nil
	case *types.Pointer:
		return find(obj, T.Elem(), append(path, opElem))
	case *types.Slice:
		return find(obj, T.Elem(), append(path, opElem))
	case *types.Array:
		return find(obj, T.Elem(), append(path, opElem))
	case *types.Chan:
		return find(obj, T.Elem(), append(path, opElem))
	case *types.Map:
		if r := find(obj, T.Key(), append(path, opKey)); r != nil {
			return r
		}
		return find(obj, T.Elem(), append(path, opElem))
	case *types.Signature:
		if r := find(obj, T.Params(), append(path, opParams)); r != nil {
			return r
		}
		return find(obj, T.Results(), append(path, opResults))
	case *types.Struct:
		for i := 0; i < T.NumFields(); i++ {
			f := T.Field(i)
			path2 := appendOpArg(path, opField, i)
			if f == obj {
				return path2 // found field var
			}
			if r := find(obj, f.Type(), append(path2, opType)); r != nil {
				return r
			}
		}
		return nil
	case *types.Tuple:
		for i := 0; i < T.Len(); i++ {
			v := T.At(i)
			path2 := appendOpArg(path, opAt, i)
			if v == obj {
				return path2 // found param/result var
			}
			if r := find(obj, v.Type(), append(path2, opType)); r != nil {
				return r
			}
		}
		return nil
	case *types.Interface:
		for i := 0; i < T.NumMethods(); i++ {
			m := T.Method(i)
			path2 := appendOpArg(path, opMethod, i)
			if m == obj {
				return path2 // found interface method
			}
			if r := find(obj, m.Type(), append(path2, opType)); r != nil {
				return r
			}
		}
		return nil
	}
	panic(T)
}

// Object returns the object denoted by path p within the package pkg.
func Object(pkg *types.Package, p Path) (types.Object, error) {
	if p == "" {
		return nil, fmt.Errorf("empty path")
	}

	pathstr := string(p)
	var pkgobj, suffix string
	if dot := strings.IndexByte(pathstr, opType); dot < 0 {
		pkgobj = pathstr
	} else {
		pkgobj = pathstr[:dot]
		suffix = pathstr[dot:] // suffix starts with "."
	}

	obj := pkg.Scope().Lookup(pkgobj)
	if obj == nil {
		return nil, fmt.Errorf("package %s does not contain %q", pkg.Path(), pkgobj)
	}

	// abtraction of *types.{Pointer,Slice,Array,Chan,Map}
	type hasElem interface {
		Elem() types.Type
	}
	// abstraction of *types.{Interface,Named}
	type hasMethods interface {
		Method(int) *types.Func
		NumMethods() int
	}

	// The loop state is the pair (t, obj),
	// exactly one of which is non-nil, initially obj.
	// All suffixes start with '.' (the only object->type operation),
	// followed by optional type->type operations,
	// then a type->object operation.
	// The cycle then repeats.
	var t types.Type
	for suffix != "" {
		code := suffix[0]
		suffix = suffix[1:]

		// Codes [AFM] have an integer operand.
		var index int
		switch code {
		case opAt, opField, opMethod:
			rest := strings.TrimLeft(suffix, "0123456789")
			numerals := suffix[:len(suffix)-len(rest)]
			suffix = rest
			i, err := strconv.Atoi(numerals)
			if err != nil {
				return nil, fmt.Errorf("invalid path: bad numeric operand %q for code %q", numerals, code)
			}
			index = int(i)
		case opObj:
			// no operand
		default:
			// The suffix must end with a type->object operation.
			if suffix == "" {
				return nil, fmt.Errorf("invalid path: ends with %q, want [AFMO]", code)
			}
		}

		if code == opType {
			if t != nil {
				return nil, fmt.Errorf("invalid path: unexpected %q in type context", opType)
			}
			t = obj.Type()
			obj = nil
			continue
		}

		if t == nil {
			return nil, fmt.Errorf("invalid path: code %q in object context", code)
		}

		// Inv: t != nil, obj == nil

		switch code {
		case opElem:
			hasElem, ok := t.(hasElem) // Pointer, Slice, Array, Chan, Map
			if !ok {
				return nil, fmt.Errorf("cannot apply %q to %s (got %T, want pointer, slice, array, chan or map)", code, t, t)
			}
			t = hasElem.Elem()

		case opKey:
			mapType, ok := t.(*types.Map)
			if !ok {
				return nil, fmt.Errorf("cannot apply %q to %s (got %T, want map)", code, t, t)
			}
			t = mapType.Key()

		case opParams:
			sig, ok := t.(*types.Signature)
			if !ok {
				return nil, fmt.Errorf("cannot apply %q to %s (got %T, want signature)", code, t, t)
			}
			t = sig.Params()

		case opResults:
			sig, ok := t.(*types.Signature)
			if !ok {
				return nil, fmt.Errorf("cannot apply %q to %s (got %T, want signature)", code, t, t)
			}
			t = sig.Results()

		case opUnderlying:
			named, ok := t.(*types.Named)
			if !ok {
				return nil, fmt.Errorf("cannot apply %q to %s (got %s, want named)", code, t, t)
			}
			t = named.Underlying()

		case opAt:
			tuple, ok := t.(*types.Tuple)
			if !ok {
				return nil, fmt.Errorf("cannot apply %q to %s (got %s, want tuple)", code, t, t)
			}
			if n := tuple.Len(); index >= n {
				return nil, fmt.Errorf("tuple index %d out of range [0-%d)", index, n)
			}
			obj = tuple.At(index)
			t = nil

		case opField:
			structType, ok := t.(*types.Struct)
			if !ok {
				return nil, fmt.Errorf("cannot apply %q to %s (got %T, want struct)", code, t, t)
			}
			if n := structType.NumFields(); index >= n {
				return nil, fmt.Errorf("field index %d out of range [0-%d)", index, n)
			}
			obj = structType.Field(index)
			t = nil

		case opMethod:
			hasMethods, ok := t.(hasMethods) // Interface or Named
			if !ok {
				return nil, fmt.Errorf("cannot apply %q to %s (got %s, want interface or named)", code, t, t)
			}
			if n := hasMethods.NumMethods(); index >= n {
				return nil, fmt.Errorf("method index %d out of range [0-%d)", index, n)
			}
			obj = hasMethods.Method(index)
			t = nil

		case opObj:
			named, ok := t.(*types.Named)
			if !ok {
				return nil, fmt.Errorf("cannot apply %q to %s (got %s, want named)", code, t, t)
			}
			obj = named.Obj()
			t = nil

		default:
			return nil, fmt.Errorf("invalid path: unknown code %q", code)
		}
	}

	if obj.Pkg() != pkg {
		return nil, fmt.Errorf("path denotes %s, which belongs to a different package", obj)
	}

	return obj, nil // success
}
