// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// This file defines a number of miscellaneous utility functions.

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"io"
	"os"
	"sync"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/typeparams"
)

//// Sanity checking utilities

// assert panics with the mesage msg if p is false.
// Avoid combining with expensive string formatting.
func assert(p bool, msg string) {
	if !p {
		panic(msg)
	}
}

//// AST utilities

func unparen(e ast.Expr) ast.Expr { return astutil.Unparen(e) }

// isBlankIdent returns true iff e is an Ident with name "_".
// They have no associated types.Object, and thus no type.
func isBlankIdent(e ast.Expr) bool {
	id, ok := e.(*ast.Ident)
	return ok && id.Name == "_"
}

//// Type utilities.  Some of these belong in go/types.

// isPointer returns true for types whose underlying type is a pointer.
func isPointer(typ types.Type) bool {
	_, ok := typ.Underlying().(*types.Pointer)
	return ok
}

func isInterface(T types.Type) bool { return types.IsInterface(T) }

// deref returns a pointer's element type; otherwise it returns typ.
func deref(typ types.Type) types.Type {
	if p, ok := typ.Underlying().(*types.Pointer); ok {
		return p.Elem()
	}
	return typ
}

// recvType returns the receiver type of method obj.
func recvType(obj *types.Func) types.Type {
	return obj.Type().(*types.Signature).Recv().Type()
}

// isUntyped returns true for types that are untyped.
func isUntyped(typ types.Type) bool {
	b, ok := typ.(*types.Basic)
	return ok && b.Info()&types.IsUntyped != 0
}

// logStack prints the formatted "start" message to stderr and
// returns a closure that prints the corresponding "end" message.
// Call using 'defer logStack(...)()' to show builder stack on panic.
// Don't forget trailing parens!
func logStack(format string, args ...interface{}) func() {
	msg := fmt.Sprintf(format, args...)
	io.WriteString(os.Stderr, msg)
	io.WriteString(os.Stderr, "\n")
	return func() {
		io.WriteString(os.Stderr, msg)
		io.WriteString(os.Stderr, " end\n")
	}
}

// newVar creates a 'var' for use in a types.Tuple.
func newVar(name string, typ types.Type) *types.Var {
	return types.NewParam(token.NoPos, nil, name, typ)
}

// anonVar creates an anonymous 'var' for use in a types.Tuple.
func anonVar(typ types.Type) *types.Var {
	return newVar("", typ)
}

var lenResults = types.NewTuple(anonVar(tInt))

// makeLen returns the len builtin specialized to type func(T)int.
func makeLen(T types.Type) *Builtin {
	lenParams := types.NewTuple(anonVar(T))
	return &Builtin{
		name: "len",
		sig:  types.NewSignature(nil, lenParams, lenResults, false),
	}
}

// nonbasicTypes returns a list containing all of the types T in ts that are non-basic.
func nonbasicTypes(ts []types.Type) []types.Type {
	if len(ts) == 0 {
		return nil
	}
	added := make(map[types.Type]bool) // additionally filter duplicates
	var filtered []types.Type
	for _, T := range ts {
		if _, basic := T.(*types.Basic); !basic {
			if !added[T] {
				added[T] = true
				filtered = append(filtered, T)
			}
		}
	}
	return filtered
}

// isGeneric returns true if a package-level member is generic.
func isGeneric(m Member) bool {
	switch m := m.(type) {
	case *NamedConst, *Global:
		return false
	case *Type:
		// lifted from types.isGeneric.
		named, _ := m.Type().(*types.Named)
		return named != nil && named.Obj() != nil && typeparams.NamedTypeArgs(named) == nil && typeparams.ForNamed(named) != nil
	case *Function:
		return len(m._TypeParams) != len(m._TypeArgs)
	default:
		panic("unreachable")
	}
}

// receiverTypeArgs returns the type arguments to a function's reciever.
// Returns an empty list if obj does not have a reciever or its reciever does not have type arguments.
func receiverTypeArgs(obj *types.Func) []types.Type {
	rtype := recvType(obj)
	if rtype == nil {
		return nil
	}
	if isPointer(rtype) {
		rtype = rtype.(*types.Pointer).Elem()
	}
	named, ok := rtype.(*types.Named)
	if !ok {
		return nil
	}
	ts := typeparams.NamedTypeArgs(named)
	if ts.Len() == 0 {
		return nil
	}
	targs := make([]types.Type, ts.Len())
	for i := 0; i < ts.Len(); i++ {
		targs[i] = ts.At(i)
	}
	return targs
}

// recvAsFirstArg takes a method signature and returns a function
// signature with receiver as the first parameter.
func recvAsFirstArg(sig *types.Signature) *types.Signature {
	params := make([]*types.Var, 0, 1+sig.Params().Len())
	params = append(params, sig.Recv())
	for i := 0; i < sig.Params().Len(); i++ {
		params = append(params, sig.Params().At(i))
	}
	return typeparams.NewSignatureType(nil, nil, nil, types.NewTuple(params...), sig.Results(), sig.Variadic())
}

// instanceArgs returns the Instance[id].TypeArgs as a slice.
func instanceArgs(info *types.Info, id *ast.Ident) []types.Type {
	targList := typeparams.GetInstances(info)[id].TypeArgs
	if targList == nil {
		return nil
	}

	targs := make([]types.Type, targList.Len())
	for i, n := 0, targList.Len(); i < n; i++ {
		targs[i] = targList.At(i)
	}
	return targs
}

// Mapping of a type T to a canonical instance C s.t. types.Indentical(T, C).
// Thread-safe.
type canonizer struct {
	mu    sync.Mutex
	types typeutil.Map // map from type to a canonical instance
	lists typeListMap  // map from a list of types to a canonical instance
}

func newCanonizer() *canonizer {
	c := &canonizer{}
	h := typeutil.MakeHasher()
	c.types.SetHasher(h)
	c.lists.hasher = h
	return c
}

// List returns a canonical representative of a list of types.
// Representative of the empty list is nil.
func (c *canonizer) List(ts []types.Type) *typeList {
	if len(ts) == 0 {
		return nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	return c.lists.rep(ts)
}

// Type returns a canonical representative of type T.
func (c *canonizer) Type(T types.Type) types.Type {
	c.mu.Lock()
	defer c.mu.Unlock()

	if r := c.types.At(T); r != nil {
		return r.(types.Type)
	}
	c.types.Set(T, T)
	return T
}

// A type for representating an canonized list of types.
type typeList []types.Type

func (l *typeList) identical(ts []types.Type) bool {
	if l == nil {
		return len(ts) == 0
	}
	n := len(*l)
	if len(ts) != n {
		return false
	}
	for i, left := range *l {
		right := ts[i]
		if !types.Identical(left, right) {
			return false
		}
	}
	return true
}

type typeListMap struct {
	hasher  typeutil.Hasher
	buckets map[uint32][]*typeList
}

// rep returns a canonical representative of a slice of types.
func (m *typeListMap) rep(ts []types.Type) *typeList {
	if m == nil || len(ts) == 0 {
		return nil
	}

	if m.buckets == nil {
		m.buckets = make(map[uint32][]*typeList)
	}

	h := m.hash(ts)
	bucket := m.buckets[h]
	for _, l := range bucket {
		if l.identical(ts) {
			return l
		}
	}

	// not present. create a representative.
	cp := make(typeList, len(ts))
	copy(cp, ts)
	rep := &cp

	m.buckets[h] = append(bucket, rep)
	return rep
}

func (m *typeListMap) hash(ts []types.Type) uint32 {
	if m == nil {
		return 0
	}
	// Some smallish prime far away from typeutil.Hash.
	n := len(ts)
	h := uint32(13619) + 2*uint32(n)
	for i := 0; i < n; i++ {
		h += 3 * m.hasher.Hash(ts[i])
	}
	return h
}

// instantiateMethod instantiates m with targs and returns a canonical representative for this method.
func (canon *canonizer) instantiateMethod(m *types.Func, targs []types.Type, ctxt *typeparams.Context) *types.Func {
	recv := recvType(m)
	if p, ok := recv.(*types.Pointer); ok {
		recv = p.Elem()
	}
	named := recv.(*types.Named)
	inst, err := typeparams.Instantiate(ctxt, typeparams.NamedTypeOrigin(named), targs, false)
	if err != nil {
		panic(err)
	}
	rep := canon.Type(inst)
	obj, _, _ := types.LookupFieldOrMethod(rep, true, m.Pkg(), m.Name())
	return obj.(*types.Func)
}
