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
//
//	type A struct{ X int }
//	type B A
//
// the field X has two paths due to its membership of both A and B.
// The For(obj) function always returns one of these paths, arbitrarily
// but consistently.
package objectpath

import (
	"fmt"
	"go/types"
	"strconv"
	"strings"

	"golang.org/x/tools/internal/typeparams"
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
//	PO package->object	Package.Scope.Lookup
//	OT  object->type 	Object.Type
//	TT    type->type 	Type.{Elem,Key,Params,Results,Underlying} [EKPRU]
//	TO   type->object	Type.{At,Field,Method,Obj} [AFMO]
//
// All valid paths start with a package and end at an object
// and thus may be defined by the regular language:
//
//	objectpath = PO (OT TT* TO)*
//
// The concrete encoding follows directly:
//   - The only PO operator is Package.Scope.Lookup, which requires an identifier.
//   - The only OT operator is Object.Type,
//     which we encode as '.' because dot cannot appear in an identifier.
//   - The TT operators are encoded as [EKPRUTC];
//     one of these (TypeParam) requires an integer operand,
//     which is encoded as a string of decimal digits.
//   - The TO operators are encoded as [AFMO];
//     three of these (At,Field,Method) require an integer operand,
//     which is encoded as a string of decimal digits.
//     These indices are stable across different representations
//     of the same package, even source and export data.
//     The indices used are implementation specific and may not correspond to
//     the argument to the go/types function.
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
//	p.Lookup("T")					T
//	.Type().Underlying().Method(0).			f
//	.Type().Results().At(1)				b
//	.Type().Field(0)					X
//
// The encoding is not maximally compact---every R or P is
// followed by an A, for example---but this simplifies the
// encoder and decoder.
const (
	// object->type operators
	opType = '.' // .Type()		  (Object)

	// type->type operators
	opElem       = 'E' // .Elem()		        (Pointer, Slice, Array, Chan, Map)
	opKey        = 'K' // .Key()		        (Map)
	opParams     = 'P' // .Params()		      (Signature)
	opResults    = 'R' // .Results()	      (Signature)
	opUnderlying = 'U' // .Underlying()	    (Named)
	opTypeParam  = 'T' // .TypeParams.At(i) (Named, Signature)
	opConstraint = 'C' // .Constraint()     (TypeParam)

	// type->object operators
	opAt     = 'A' // .At(i)		 (Tuple)
	opField  = 'F' // .Field(i)	 (Struct)
	opMethod = 'M' // .Method(i) (Named or Interface; not Struct: "promoted" names are ignored)
	opObj    = 'O' // .Obj()		 (Named, TypeParam)
)

// For is equivalent to new(Encoder).For(obj).
//
// It may be more efficient to reuse a single Encoder across several calls.
func For(obj types.Object) (Path, error) {
	return new(Encoder).For(obj)
}

// An Encoder amortizes the cost of encoding the paths of multiple objects.
// The zero value of an Encoder is ready to use.
type Encoder struct {
	scopeMemo map[*types.Scope][]types.Object // memoization of scopeObjects
}

// For returns the path to an object relative to its package,
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
// The set of objects accessible from a package's Scope depends on
// whether the package was produced by type-checking syntax, or
// reading export data; the latter may have a smaller Scope since
// export data trims objects that are not reachable from an exported
// declaration. For example, the For function will return a path for
// an exported method of an unexported type that is not reachable
// from any public declaration; this path will cause the Object
// function to fail if called on a package loaded from export data.
// TODO(adonovan): is this a bug or feature? Should this package
// compute accessibility in the same way?
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
//	p.Scope().Lookup("T")				(TypeName T)
//	.Type().Underlying().Method(0).			(method Func f)
//	.Type().Results().At(1)				(field Var b)
//	.Type().Field(0)					(field Var X)
//
// where p is the package (*types.Package) to which X belongs.
func (enc *Encoder) For(obj types.Object) (Path, error) {
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
	case *types.TypeName:
		if _, ok := obj.Type().(*types.TypeParam); !ok {
			// With the exception of type parameters, only package-level type names
			// have a path.
			return "", fmt.Errorf("no path for %v", obj)
		}
	case *types.Const, // Only package-level constants have a path.
		*types.Label,   // Labels are function-local.
		*types.PkgName: // PkgNames are file-local.
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

		if path, ok := enc.concreteMethod(obj); ok {
			// Fast path for concrete methods that avoids looping over scope.
			return path, nil
		}

	default:
		panic(obj)
	}

	// 4. Search the API for the path to the var (field/param/result) or method.

	// First inspect package-level named types.
	// In the presence of path aliases, these give
	// the best paths because non-types may
	// refer to types, but not the reverse.
	empty := make([]byte, 0, 48) // initial space
	objs := enc.scopeObjects(scope)
	for _, o := range objs {
		tname, ok := o.(*types.TypeName)
		if !ok {
			continue // handle non-types in second pass
		}

		path := append(empty, o.Name()...)
		path = append(path, opType)

		T := o.Type()

		if tname.IsAlias() {
			// type alias
			if r := find(obj, T, path, nil); r != nil {
				return Path(r), nil
			}
		} else {
			if named, _ := T.(*types.Named); named != nil {
				if r := findTypeParam(obj, named.TypeParams(), path, nil); r != nil {
					// generic named type
					return Path(r), nil
				}
			}
			// defined (named) type
			if r := find(obj, T.Underlying(), append(path, opUnderlying), nil); r != nil {
				return Path(r), nil
			}
		}
	}

	// Then inspect everything else:
	// non-types, and declared methods of defined types.
	for _, o := range objs {
		path := append(empty, o.Name()...)
		if _, ok := o.(*types.TypeName); !ok {
			if o.Exported() {
				// exported non-type (const, var, func)
				if r := find(obj, o.Type(), append(path, opType), nil); r != nil {
					return Path(r), nil
				}
			}
			continue
		}

		// Inspect declared methods of defined types.
		if T, ok := o.Type().(*types.Named); ok {
			path = append(path, opType)
			// The method index here is always with respect
			// to the underlying go/types data structures,
			// which ultimately derives from source order
			// and must be preserved by export data.
			for i := 0; i < T.NumMethods(); i++ {
				m := T.Method(i)
				path2 := appendOpArg(path, opMethod, i)
				if m == obj {
					return Path(path2), nil // found declared method
				}
				if r := find(obj, m.Type(), append(path2, opType), nil); r != nil {
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

// concreteMethod returns the path for meth, which must have a non-nil receiver.
// The second return value indicates success and may be false if the method is
// an interface method or if it is an instantiated method.
//
// This function is just an optimization that avoids the general scope walking
// approach. You are expected to fall back to the general approach if this
// function fails.
func (enc *Encoder) concreteMethod(meth *types.Func) (Path, bool) {
	// Concrete methods can only be declared on package-scoped named types. For
	// that reason we can skip the expensive walk over the package scope: the
	// path will always be package -> named type -> method. We can trivially get
	// the type name from the receiver, and only have to look over the type's
	// methods to find the method index.
	//
	// Methods on generic types require special consideration, however. Consider
	// the following package:
	//
	// 	L1: type S[T any] struct{}
	// 	L2: func (recv S[A]) Foo() { recv.Bar() }
	// 	L3: func (recv S[B]) Bar() { }
	// 	L4: type Alias = S[int]
	// 	L5: func _[T any]() { var s S[int]; s.Foo() }
	//
	// The receivers of methods on generic types are instantiations. L2 and L3
	// instantiate S with the type-parameters A and B, which are scoped to the
	// respective methods. L4 and L5 each instantiate S with int. Each of these
	// instantiations has its own method set, full of methods (and thus objects)
	// with receivers whose types are the respective instantiations. In other
	// words, we have
	//
	// S[A].Foo, S[A].Bar
	// S[B].Foo, S[B].Bar
	// S[int].Foo, S[int].Bar
	//
	// We may thus be trying to produce object paths for any of these objects.
	//
	// S[A].Foo and S[B].Bar are the origin methods, and their paths are S.Foo
	// and S.Bar, which are the paths that this function naturally produces.
	//
	// S[A].Bar, S[B].Foo, and both methods on S[int] are instantiations that
	// don't correspond to the origin methods. For S[int], this is significant.
	// The most precise object path for S[int].Foo, for example, is Alias.Foo,
	// not S.Foo. Our function, however, would produce S.Foo, which would
	// resolve to a different object.
	//
	// For S[A].Bar and S[B].Foo it could be argued that S.Bar and S.Foo are
	// still the correct paths, since only the origin methods have meaningful
	// paths. But this is likely only true for trivial cases and has edge cases.
	// Since this function is only an optimization, we err on the side of giving
	// up, deferring to the slower but definitely correct algorithm. Most users
	// of objectpath will only be giving us origin methods, anyway, as referring
	// to instantiated methods is usually not useful.

	if typeparams.OriginMethod(meth) != meth {
		return "", false
	}

	recvT := meth.Type().(*types.Signature).Recv().Type()
	if ptr, ok := recvT.(*types.Pointer); ok {
		recvT = ptr.Elem()
	}

	named, ok := recvT.(*types.Named)
	if !ok {
		return "", false
	}

	if types.IsInterface(named) {
		// Named interfaces don't have to be package-scoped
		//
		// TODO(dominikh): opt: if scope.Lookup(name) == named, then we can apply this optimization to interface
		// methods, too, I think.
		return "", false
	}

	// Preallocate space for the name, opType, opMethod, and some digits.
	name := named.Obj().Name()
	path := make([]byte, 0, len(name)+8)
	path = append(path, name...)
	path = append(path, opType)

	// Method indices are w.r.t. the go/types data structures,
	// ultimately deriving from source order,
	// which is preserved by export data.
	for i := 0; i < named.NumMethods(); i++ {
		if named.Method(i) == meth {
			path = appendOpArg(path, opMethod, i)
			return Path(path), true
		}
	}

	// Due to golang/go#59944, go/types fails to associate the receiver with
	// certain methods on cgo types.
	//
	// TODO(rfindley): replace this panic once golang/go#59944 is fixed in all Go
	// versions gopls supports.
	return "", false
	// panic(fmt.Sprintf("couldn't find method %s on type %s; methods: %#v", meth, named, enc.namedMethods(named)))
}

// find finds obj within type T, returning the path to it, or nil if not found.
//
// The seen map is used to short circuit cycles through type parameters. If
// nil, it will be allocated as necessary.
func find(obj types.Object, T types.Type, path []byte, seen map[*types.TypeName]bool) []byte {
	switch T := T.(type) {
	case *types.Basic, *types.Named:
		// Named types belonging to pkg were handled already,
		// so T must belong to another package. No path.
		return nil
	case *types.Pointer:
		return find(obj, T.Elem(), append(path, opElem), seen)
	case *types.Slice:
		return find(obj, T.Elem(), append(path, opElem), seen)
	case *types.Array:
		return find(obj, T.Elem(), append(path, opElem), seen)
	case *types.Chan:
		return find(obj, T.Elem(), append(path, opElem), seen)
	case *types.Map:
		if r := find(obj, T.Key(), append(path, opKey), seen); r != nil {
			return r
		}
		return find(obj, T.Elem(), append(path, opElem), seen)
	case *types.Signature:
		if r := findTypeParam(obj, T.TypeParams(), path, seen); r != nil {
			return r
		}
		if r := find(obj, T.Params(), append(path, opParams), seen); r != nil {
			return r
		}
		return find(obj, T.Results(), append(path, opResults), seen)
	case *types.Struct:
		for i := 0; i < T.NumFields(); i++ {
			fld := T.Field(i)
			path2 := appendOpArg(path, opField, i)
			if fld == obj {
				return path2 // found field var
			}
			if r := find(obj, fld.Type(), append(path2, opType), seen); r != nil {
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
			if r := find(obj, v.Type(), append(path2, opType), seen); r != nil {
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
			if r := find(obj, m.Type(), append(path2, opType), seen); r != nil {
				return r
			}
		}
		return nil
	case *types.TypeParam:
		name := T.Obj()
		if name == obj {
			return append(path, opObj)
		}
		if seen[name] {
			return nil
		}
		if seen == nil {
			seen = make(map[*types.TypeName]bool)
		}
		seen[name] = true
		if r := find(obj, T.Constraint(), append(path, opConstraint), seen); r != nil {
			return r
		}
		return nil
	}
	panic(T)
}

func findTypeParam(obj types.Object, list *types.TypeParamList, path []byte, seen map[*types.TypeName]bool) []byte {
	for i := 0; i < list.Len(); i++ {
		tparam := list.At(i)
		path2 := appendOpArg(path, opTypeParam, i)
		if r := find(obj, tparam, path2, seen); r != nil {
			return r
		}
	}
	return nil
}

// Object returns the object denoted by path p within the package pkg.
func Object(pkg *types.Package, p Path) (types.Object, error) {
	pathstr := string(p)
	if pathstr == "" {
		return nil, fmt.Errorf("empty path")
	}

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

	// abstraction of *types.{Pointer,Slice,Array,Chan,Map}
	type hasElem interface {
		Elem() types.Type
	}
	// abstraction of *types.{Named,Signature}
	type hasTypeParams interface {
		TypeParams() *types.TypeParamList
	}
	// abstraction of *types.{Named,TypeParam}
	type hasObj interface {
		Obj() *types.TypeName
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
		case opAt, opField, opMethod, opTypeParam:
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
				return nil, fmt.Errorf("cannot apply %q to %s (got %T, want named)", code, t, t)
			}
			t = named.Underlying()

		case opTypeParam:
			hasTypeParams, ok := t.(hasTypeParams) // Named, Signature
			if !ok {
				return nil, fmt.Errorf("cannot apply %q to %s (got %T, want named or signature)", code, t, t)
			}
			tparams := hasTypeParams.TypeParams()
			if n := tparams.Len(); index >= n {
				return nil, fmt.Errorf("tuple index %d out of range [0-%d)", index, n)
			}
			t = tparams.At(index)

		case opConstraint:
			tparam, ok := t.(*types.TypeParam)
			if !ok {
				return nil, fmt.Errorf("cannot apply %q to %s (got %T, want type parameter)", code, t, t)
			}
			t = tparam.Constraint()

		case opAt:
			tuple, ok := t.(*types.Tuple)
			if !ok {
				return nil, fmt.Errorf("cannot apply %q to %s (got %T, want tuple)", code, t, t)
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
			switch t := t.(type) {
			case *types.Interface:
				if index >= t.NumMethods() {
					return nil, fmt.Errorf("method index %d out of range [0-%d)", index, t.NumMethods())
				}
				obj = t.Method(index) // Id-ordered

			case *types.Named:
				if index >= t.NumMethods() {
					return nil, fmt.Errorf("method index %d out of range [0-%d)", index, t.NumMethods())
				}
				obj = t.Method(index)

			default:
				return nil, fmt.Errorf("cannot apply %q to %s (got %T, want interface or named)", code, t, t)
			}
			t = nil

		case opObj:
			hasObj, ok := t.(hasObj)
			if !ok {
				return nil, fmt.Errorf("cannot apply %q to %s (got %T, want named or type param)", code, t, t)
			}
			obj = hasObj.Obj()
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

// scopeObjects is a memoization of scope objects.
// Callers must not modify the result.
func (enc *Encoder) scopeObjects(scope *types.Scope) []types.Object {
	m := enc.scopeMemo
	if m == nil {
		m = make(map[*types.Scope][]types.Object)
		enc.scopeMemo = m
	}
	objs, ok := m[scope]
	if !ok {
		names := scope.Names() // allocates and sorts
		objs = make([]types.Object, len(names))
		for i, name := range names {
			objs[i] = scope.Lookup(name)
		}
		m[scope] = objs
	}
	return objs
}
