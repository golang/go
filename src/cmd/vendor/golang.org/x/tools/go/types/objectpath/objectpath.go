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
	"encoding/binary"
	"fmt"
	"go/types"
	"slices"
	"strconv"
	"strings"

	"golang.org/x/tools/internal/typesinternal"
)

// TODO(adonovan): think about generic aliases.

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
//	TT    type->type 	Type.{Elem,Key,{,{,Recv}Type}Params,Results,Underlying,Rhs} [EKPRUTrCa]
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
//   - The TT operators are encoded as [EKPRUTrCa];
//     two of these ({,Recv}TypeParams) require an integer operand,
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
	opElem          = 'E' // .Elem()		(Pointer, Slice, Array, Chan, Map)
	opKey           = 'K' // .Key()			(Map)
	opParams        = 'P' // .Params()		(Signature)
	opResults       = 'R' // .Results()		(Signature)
	opUnderlying    = 'U' // .Underlying()		(Named)
	opTypeParam     = 'T' // .TypeParams.At(i)	(Named, Signature)
	opRecvTypeParam = 'r' // .RecvTypeParams.At(i)	(Signature)
	opConstraint    = 'C' // .Constraint()		(TypeParam)
	opRhs           = 'a' // .Rhs()			(Alias)

	// type->object operators
	opAt     = 'A' // .At(i)	(Tuple)
	opField  = 'F' // .Field(i)	(Struct)
	opMethod = 'M' // .Method(i)	(Named or Interface; not Struct: "promoted" names are ignored)
	opObj    = 'O' // .Obj()	(Named, TypeParam)
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
	pkgIndex map[*types.Package]*pkgIndex
}

// A traversal encapsulates the state of a single traversal of the object/type graph.
type traversal struct {
	pkg *types.Package
	ix  *pkgIndex // non-nil if we are building the index

	target types.Object // the sought symbol (if ix == nil)
	found  Path         // the found path    (if ix == nil)

	// These maps are used to short circuit cycles through
	// interface methods, such as occur in the following example:
	//
	//	type I interface { f() interface{I} }
	//
	// See golang/go#68046 for details.
	seenTParamNames map[*types.TypeName]bool // global cycle breaking through type parameters
	seenMethods     map[*types.Func]bool     // global cycle breaking through recursive interfaces
}

// A pkgIndex holds a compressed index of objectpaths of all symbols
// (fields, methods, params) requiring search for an entire package.
//
// The first time a search for a given package is requested, we simply
// traverse the type graph for the target object, maintaining the
// current object path as a stack. If we find the target object, we
// save the path and terminate the main loop (but it's not worth
// breaking out of the current recursion).
//
// On the second search (a pkgIndex exists but its data is nil), we
// build an index of the traversal, which we use for all subsequent
// searches.
//
// The traversal index is encoded in the data field as a list of records,
// one per node, in preorder. Records are of two types:
//
//   - A record for a package-level object consists of a pair
//     (parent, nameIndex uvarint), where parent is zero and
//     nameIndex is the index of the object's name in the sorted
//     pkg.Scope().Names() slice.
//
//   - A record for a nested node (a segment of an object path)
//     consists of (parent uvarint, op byte, index uvarint), where
//     parent is the index of the record for the parent node,
//     op is the destructuring operator, and index (if op = [AFMTr])
//     is its integer operand.
//
// Since data[0] = 0 all nodes have positive offsets. In effect the
// encoding is a trie in which each node stores one path segment
// and points to the node for its prefix.
//
// TODO(adonovan): opt: evaluate an only 2-level tree with nodes for
// package-level objects and the-rest-of-the-path. One calculation
// suggested that it might be similar speed but 30% more compact.
type pkgIndex struct {
	pkg        *types.Package
	data       []byte                  // encoding of traversal; nil if not yet constructed
	scopeNames []string                // memo of pkg.Scope().Names() to avoid O(n) alloc/sort at lookup
	offsets    map[types.Object]uint32 // each object's node offset within encoded traversal data
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

	// 2. package-level object?
	if pkg.Scope().Lookup(obj.Name()) == obj {
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
		if _, ok := types.Unalias(obj.Type()).(*types.TypeParam); !ok {
			// With the exception of type parameters, only package-level type names
			// have a path.
			return "", fmt.Errorf("no path for %v", obj)
		}

	case *types.Const, // Only package-level constants have a path.
		*types.Label,   // Labels are function-local.
		*types.PkgName: // PkgNames are file-local.
		return "", fmt.Errorf("no path for %v", obj)

	case *types.Var:
		// A var, if not package-level, must be a
		// parameter (incl. receiver) or result, or a struct field.
		if obj.Kind() == types.LocalVar {
			return "", fmt.Errorf("no path for local %v", obj)
		}

	case *types.Func:
		// A func, if not package-level, must be a method.
		if recv := obj.Signature().Recv(); recv == nil {
			return "", fmt.Errorf("func is not a method: %v", obj)
		}

		if path, ok := enc.concreteMethod(obj); ok {
			// Fast path for concrete methods that avoids looping over scope.
			return path, nil
		}

	default:
		panic(obj)
	}

	// 4. Search the object/type graph for the path to
	//    the var (field/param/result) or method.
	ix, ok := enc.pkgIndex[pkg]
	if !ok {
		// First search: don't build an index, just traverse.
		// This avoids allocation in [For], whose Encoder
		// lives for a single call.
		ix = &pkgIndex{pkg: pkg}

		if enc.pkgIndex == nil {
			enc.pkgIndex = make(map[*types.Package]*pkgIndex)
		}
		enc.pkgIndex[pkg] = ix // build the index next time

		f := traversal{pkg: pkg, target: obj}
		f.traverse()

		if f.found != "" {
			return f.found, nil
		}
	} else {
		// Second search: build an index while traversing.
		if ix.data == nil {
			ix.offsets = make(map[types.Object]uint32)
			ix.data = []byte{0} // offset 0 is sentinel
			(&traversal{pkg: pkg, ix: ix}).traverse()
		}

		// Second and later searches: consult the index.
		if offset, ok := ix.offsets[obj]; ok {
			return ix.path(offset), nil
		}
	}

	return "", fmt.Errorf("can't find path for %v in %s", obj, pkg.Path())
}

// traverse performs a complete traversal of all symbols reachable from the package.
func (tr *traversal) traverse() {
	scope := tr.pkg.Scope()
	names := scope.Names()
	if tr.ix != nil {
		tr.ix.scopeNames = names
	}

	empty := make([]byte, 0, 48) // initial space for stack (ix == nil)

	// First inspect package-level type names.
	// In the presence of path aliases, these give
	// the best paths because non-types may
	// refer to types, but not the reverse.
	for i, name := range names {
		if tr.found != "" {
			return // found (ix == nil)
		}

		obj := scope.Lookup(name)
		if _, ok := obj.(*types.TypeName); !ok {
			continue // handle non-types in second pass
		}

		// emit (name, opType)
		var path []byte
		var offset uint32
		if tr.ix == nil {
			path = append(empty, name...)
			path = append(path, opType)
		} else {
			offset = tr.ix.emitPackageLevel(i)
			tr.ix.offsets[obj] = offset
			offset = tr.ix.emitPathSegment(offset, opType, -1)
		}

		// A TypeName (for Named or Alias) may have type parameters.
		switch t := obj.Type().(type) {
		case *types.Alias:
			tr.tparams(t.TypeParams(), path, offset, opTypeParam)
			tr.typ(path, offset, opRhs, -1, t.Rhs())
		case *types.Named:
			tr.tparams(t.TypeParams(), path, offset, opTypeParam)
			tr.typ(path, offset, opUnderlying, -1, t.Underlying())
		}
	}

	// Then inspect everything else:
	// exported non-types, and declared methods of defined types.
	for i, name := range names {
		if tr.found != "" {
			return // found (ix == nil)
		}

		obj := scope.Lookup(name)

		if tname, ok := obj.(*types.TypeName); !ok {
			if obj.Exported() {
				// exported non-type (const, var, func)
				var path []byte
				var offset uint32
				if tr.ix == nil {
					path = append(empty, name...)
				} else {
					offset = tr.ix.emitPackageLevel(i)
					tr.ix.offsets[obj] = offset
				}
				tr.typ(path, offset, opType, -1, obj.Type())
			}

		} else if T, ok := types.Unalias(tname.Type()).(*types.Named); ok {
			// defined type
			var path []byte
			var offset uint32
			if tr.ix == nil {
				path = append(empty, name...)
				path = append(path, opType)
			} else {
				// Inv: map entry for obj was populated in first pass.
				offset = tr.ix.emitPathSegment(tr.ix.offsets[obj], opType, -1)
			}

			// Inspect declared methods of defined types.
			//
			// The method index here is always with respect
			// to the underlying go/types data structures,
			// which ultimately derives from source order
			// and must be preserved by export data.
			for i := 0; i < T.NumMethods(); i++ {
				m := T.Method(i)
				tr.object(path, offset, opMethod, i, m)
			}
		}
	}
}

func (tr *traversal) visitType(path []byte, offset uint32, T types.Type) {
	switch T := T.(type) {
	case *types.Alias:
		tr.typ(path, offset, opRhs, -1, T.Rhs())

	case *types.Basic, *types.Named:
		// Named types belonging to pkg were handled already,
		// so T must belong to another package. No path.
		return

	case *types.Pointer, *types.Slice, *types.Array, *types.Chan:
		type hasElem interface{ Elem() types.Type } // note: includes Map
		tr.typ(path, offset, opElem, -1, T.(hasElem).Elem())

	case *types.Map:
		tr.typ(path, offset, opKey, -1, T.Key())
		tr.typ(path, offset, opElem, -1, T.Elem())

	case *types.Signature:
		tr.tparams(T.RecvTypeParams(), path, offset, opRecvTypeParam)
		tr.tparams(T.TypeParams(), path, offset, opTypeParam)
		tr.typ(path, offset, opParams, -1, T.Params())
		tr.typ(path, offset, opResults, -1, T.Results())

	case *types.Struct:
		for i := 0; i < T.NumFields(); i++ {
			tr.object(path, offset, opField, i, T.Field(i))
		}

	case *types.Tuple:
		for i := 0; i < T.Len(); i++ {
			tr.object(path, offset, opAt, i, T.At(i))
		}

	case *types.Interface:
		for i := 0; i < T.NumMethods(); i++ {
			m := T.Method(i)
			if m.Pkg() != nil && m.Pkg() != tr.pkg {
				continue // embedded method from another package
			}
			if !tr.seenMethods[m] {
				if tr.seenMethods == nil {
					tr.seenMethods = make(map[*types.Func]bool)
				}
				tr.seenMethods[m] = true
				tr.object(path, offset, opMethod, i, m)
			}
		}

	case *types.TypeParam:
		tname := T.Obj()
		if tname.Pkg() != nil && tname.Pkg() != tr.pkg {
			return // type parameter from another package
		}
		if !tr.seenTParamNames[tname] {
			if tr.seenTParamNames == nil {
				tr.seenTParamNames = make(map[*types.TypeName]bool)
			}
			tr.seenTParamNames[tname] = true
			tr.object(path, offset, opObj, -1, tname)
			tr.typ(path, offset, opConstraint, -1, T.Constraint())
		}
	}
}

func (tr *traversal) tparams(list *types.TypeParamList, path []byte, offset uint32, op byte) {
	for i := 0; i < list.Len(); i++ {
		tr.typ(path, offset, op, i, list.At(i))
	}
}

// typ descends the type graph edge (op, index), then proceeds to traverse type t.
func (tr *traversal) typ(path []byte, offset uint32, op byte, index int, t types.Type) {
	if tr.ix == nil {
		path = appendOpArg(path, op, index)
	} else {
		offset = tr.ix.emitPathSegment(offset, op, index)
	}
	tr.visitType(path, offset, t)
}

// object descends the type graph edge (op, index), records object
// obj, then proceeds to traverse its type.
func (tr *traversal) object(path []byte, offset uint32, op byte, index int, obj types.Object) {
	if tr.ix == nil {
		path = appendOpArg(path, op, index)
		if obj == tr.target && tr.found == "" {
			tr.found = Path(path)
		}
		path = append(path, opType)
	} else {
		offset = tr.ix.emitPathSegment(offset, op, index)
		if _, ok := tr.ix.offsets[obj]; !ok {
			tr.ix.offsets[obj] = offset
		}
		offset = tr.ix.emitPathSegment(offset, opType, -1)
	}
	tr.visitType(path, offset, obj.Type())
}

// emitPackageLevel encodes a record for a package-level symbol,
// identified by its index in ix.scopeNames.
func (p *pkgIndex) emitPackageLevel(index int) uint32 {
	off := uint32(len(p.data))
	p.data = append(p.data, 0) // zero varint => no parent
	p.data = binary.AppendUvarint(p.data, uint64(index))
	return off
}

// emitPathSegment emits a record for a non-initial object path segment.
func (p *pkgIndex) emitPathSegment(parent uint32, op byte, index int) uint32 {
	off := uint32(len(p.data))
	p.data = binary.AppendUvarint(p.data, uint64(parent))
	p.data = append(p.data, op)
	switch op {
	case opAt, opField, opMethod, opTypeParam, opRecvTypeParam:
		p.data = binary.AppendUvarint(p.data, uint64(index))
	}
	return off
}

// path returns the Path for the encoded node at the specified offset.
func (p *pkgIndex) path(offset uint32) Path {
	var elems []string // path elements in reverse
	for {
		// Read parent index.
		parent, n := binary.Uvarint(p.data[offset:])
		offset += uint32(n)

		if parent == 0 {
			break // root (end of path)
		}

		op := p.data[offset]
		offset++

		// The [AFMTr] operators have a numeric operand.
		switch op {
		case opAt, opField, opMethod, opTypeParam, opRecvTypeParam:
			val, n := binary.Uvarint(p.data[offset:])
			offset += uint32(n)
			elems = append(elems, strconv.Itoa(int(val)))
		}

		elems = append(elems, string([]byte{op}))

		offset = uint32(parent)
	}
	idx, _ := binary.Uvarint(p.data[offset:])

	// Convert index to Path string.
	name := p.scopeNames[idx]
	sz := len(name)
	for _, elem := range elems {
		sz += len(elem)
	}
	var buf strings.Builder
	buf.Grow(sz)
	buf.WriteString(name)
	for _, elem := range slices.Backward(elems) {
		buf.WriteString(elem)
	}
	return Path(buf.String())
}

// appendOpArg appends (op, index) to the object path.
// A negative index is ignored.
func appendOpArg(path []byte, op byte, index int) []byte {
	path = append(path, op)
	if index >= 0 {
		path = strconv.AppendInt(path, int64(index), 10)
	}
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

	if meth.Origin() != meth {
		return "", false
	}

	_, named := typesinternal.ReceiverNamed(meth.Signature().Recv())
	if named == nil {
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
	// abstraction of *types.{Alias,Named,TypeParam}
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

		// Codes [AFMTr] have an integer operand.
		var index int
		switch code {
		case opAt, opField, opMethod, opTypeParam, opRecvTypeParam:
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

		t = types.Unalias(t)
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

		case opRhs:
			if alias, ok := t.(*types.Alias); ok {
				t = alias.Rhs()
			} else if false {
				// Now that go1.24 is assured, we should be able to
				// replace this with "if true {", but it causes objectpath
				// tests to fail. TODO(adonovan): investigate.
				return nil, fmt.Errorf("cannot apply %q to %s (got %T, want alias)", code, t, t)
			}

		case opTypeParam:
			hasTypeParams, ok := t.(hasTypeParams) // Named, Signature
			if !ok {
				return nil, fmt.Errorf("cannot apply %q to %s (got %T, want named or signature)", code, t, t)
			}
			tparams := hasTypeParams.TypeParams()
			if n := tparams.Len(); index >= n {
				return nil, fmt.Errorf("type parameter index %d out of range [0-%d)", index, n)
			}
			t = tparams.At(index)

		case opRecvTypeParam:
			sig, ok := t.(*types.Signature) // Signature
			if !ok {
				return nil, fmt.Errorf("cannot apply %q to %s (got %T, want signature)", code, t, t)
			}
			rtparams := sig.RecvTypeParams()
			if n := rtparams.Len(); index >= n {
				return nil, fmt.Errorf("receiver type parameter index %d out of range [0-%d)", index, n)
			}
			t = rtparams.At(index)

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

	if obj == nil {
		panic(p) // path does not end in an object-valued operator
	}

	if obj.Pkg() != pkg {
		return nil, fmt.Errorf("path denotes %s, which belongs to a different package", obj)
	}

	return obj, nil // success
}
