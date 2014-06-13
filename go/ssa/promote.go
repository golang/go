// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// This file defines utilities for population of method sets and
// synthesis of Functions that delegate to declared methods, which
// come in three kinds:
//
// (1) wrappers: methods that wrap declared methods, performing
//     implicit pointer indirections and embedded field selections.
//
// (2) thunks: funcs that wrap declared methods.  Like wrappers,
//     thunks perform indirections and field selections. The thunks's
//     first parameter is used as the receiver for the method call.
//
// (3) bounds: funcs that wrap declared methods.  The bound's sole
//     free variable, supplied by a closure, is used as the receiver
//     for the method call.  No indirections or field selections are
//     performed since they can be done before the call.
//
// TODO(adonovan): split and rename to {methodset,delegate}.go.
// TODO(adonovan): use 'sel' not 'meth' for *types.Selection; reserve 'meth' for *Function.

import (
	"fmt"

	"code.google.com/p/go.tools/go/types"
)

// Method returns the Function implementing method meth, building
// wrapper methods on demand.  It returns nil if meth denotes an
// abstract (interface) method.
//
// Precondition: meth Kind() == MethodVal.
//
// TODO(adonovan): rename this to MethodValue because of the
// precondition, and for consistency with functions in source.go.
//
// Thread-safe.
//
// EXCLUSIVE_LOCKS_ACQUIRED(prog.methodsMu)
//
func (prog *Program) Method(meth *types.Selection) *Function {
	if meth.Kind() != types.MethodVal {
		panic(fmt.Sprintf("Method(%s) kind != MethodVal", meth))
	}
	T := meth.Recv()
	if isInterface(T) {
		return nil // abstract method
	}
	if prog.mode&LogSource != 0 {
		defer logStack("Method %s %v", T, meth)()
	}

	prog.methodsMu.Lock()
	defer prog.methodsMu.Unlock()

	return prog.addMethod(prog.createMethodSet(T), meth)
}

// LookupMethod returns the implementation of the method of type T
// identified by (pkg, name).  It returns nil if the method exists but
// is abstract, and panics if T has no such method.
//
func (prog *Program) LookupMethod(T types.Type, pkg *types.Package, name string) *Function {
	sel := prog.MethodSets.MethodSet(T).Lookup(pkg, name)
	if sel == nil {
		panic(fmt.Sprintf("%s has no method %s", T, types.Id(pkg, name)))
	}
	return prog.Method(sel)
}

// makeMethods ensures that all concrete methods of type T are
// generated.  It is equivalent to calling prog.Method() on all
// members of T.methodSet(), but acquires fewer locks.
//
// It reports whether the type's (concrete) method set is non-empty.
//
// Thread-safe.
//
// EXCLUSIVE_LOCKS_ACQUIRED(prog.methodsMu)
//
func (prog *Program) makeMethods(T types.Type) bool {
	if isInterface(T) {
		return false // abstract method
	}
	tmset := prog.MethodSets.MethodSet(T)
	n := tmset.Len()
	if n == 0 {
		return false // empty (common case)
	}

	if prog.mode&LogSource != 0 {
		defer logStack("makeMethods %s", T)()
	}

	prog.methodsMu.Lock()
	defer prog.methodsMu.Unlock()

	mset := prog.createMethodSet(T)
	if !mset.complete {
		mset.complete = true
		for i := 0; i < n; i++ {
			prog.addMethod(mset, tmset.At(i))
		}
	}

	return true
}

// methodSet contains the (concrete) methods of a non-interface type.
type methodSet struct {
	mapping  map[string]*Function // populated lazily
	complete bool                 // mapping contains all methods
}

// Precondition: !isInterface(T).
// EXCLUSIVE_LOCKS_REQUIRED(prog.methodsMu)
func (prog *Program) createMethodSet(T types.Type) *methodSet {
	mset, ok := prog.methodSets.At(T).(*methodSet)
	if !ok {
		mset = &methodSet{mapping: make(map[string]*Function)}
		prog.methodSets.Set(T, mset)
	}
	return mset
}

// EXCLUSIVE_LOCKS_REQUIRED(prog.methodsMu)
func (prog *Program) addMethod(mset *methodSet, sel *types.Selection) *Function {
	if sel.Kind() == types.MethodExpr {
		panic(sel)
	}
	id := sel.Obj().Id()
	fn := mset.mapping[id]
	if fn == nil {
		obj := sel.Obj().(*types.Func)

		needsPromotion := len(sel.Index()) > 1
		needsIndirection := !isPointer(recvType(obj)) && isPointer(sel.Recv())
		if needsPromotion || needsIndirection {
			fn = makeWrapper(prog, sel)
		} else {
			fn = prog.declaredFunc(obj)
		}
		if fn.Signature.Recv() == nil {
			panic(fn) // missing receiver
		}
		mset.mapping[id] = fn
	}
	return fn
}

// TypesWithMethodSets returns a new unordered slice containing all
// concrete types in the program for which a complete (non-empty)
// method set is required at run-time.
//
// It is the union of pkg.TypesWithMethodSets() for all pkg in
// prog.AllPackages().
//
// Thread-safe.
//
// EXCLUSIVE_LOCKS_ACQUIRED(prog.methodsMu)
//
func (prog *Program) TypesWithMethodSets() []types.Type {
	prog.methodsMu.Lock()
	defer prog.methodsMu.Unlock()

	var res []types.Type
	prog.methodSets.Iterate(func(T types.Type, v interface{}) {
		if v.(*methodSet).complete {
			res = append(res, T)
		}
	})
	return res
}

// TypesWithMethodSets returns an unordered slice containing the set
// of all concrete types referenced within package pkg and not
// belonging to some other package, for which a complete (non-empty)
// method set is required at run-time.
//
// A type belongs to a package if it is a named type or a pointer to a
// named type, and the name was defined in that package.  All other
// types belong to no package.
//
// A type may appear in the TypesWithMethodSets() set of multiple
// distinct packages if that type belongs to no package.  Typical
// compilers emit method sets for such types multiple times (using
// weak symbols) into each package that references them, with the
// linker performing duplicate elimination.
//
// This set includes the types of all operands of some MakeInterface
// instruction, the types of all exported members of some package, and
// all types that are subcomponents, since even types that aren't used
// directly may be derived via reflection.
//
// Callers must not mutate the result.
//
func (pkg *Package) TypesWithMethodSets() []types.Type {
	return pkg.methodSets
}

// declaredFunc returns the concrete function/method denoted by obj.
// Panic ensues if there is none.
//
func (prog *Program) declaredFunc(obj *types.Func) *Function {
	if v := prog.packageLevelValue(obj); v != nil {
		return v.(*Function)
	}
	panic("no concrete method: " + obj.String())
}

// -- wrappers ---------------------------------------------------------

// makeWrapper returns a synthetic method that delegates to the
// declared method denoted by meth.Obj(), first performing any
// necessary pointer indirections or field selections implied by meth.
//
// The resulting method's receiver type is meth.Recv().
//
// This function is versatile but quite subtle!  Consider the
// following axes of variation when making changes:
//   - optional receiver indirection
//   - optional implicit field selections
//   - meth.Obj() may denote a concrete or an interface method
//   - the result may be a thunk or a wrapper.
//
// EXCLUSIVE_LOCKS_REQUIRED(prog.methodsMu)
//
func makeWrapper(prog *Program, meth *types.Selection) *Function {
	obj := meth.Obj().(*types.Func)       // the declared function
	sig := meth.Type().(*types.Signature) // type of this wrapper

	var recv *types.Var // wrapper's receiver or thunk's params[0]
	name := obj.Name()
	var description string
	var start int // first regular param
	if meth.Kind() == types.MethodExpr {
		name += "$thunk"
		description = "thunk"
		recv = sig.Params().At(0)
		start = 1
	} else {
		description = "wrapper"
		recv = sig.Recv()
	}

	description = fmt.Sprintf("%s for %s", description, meth.Obj())
	if prog.mode&LogSource != 0 {
		defer logStack("make %s to (%s)", description, recv.Type())()
	}
	fn := &Function{
		name:      name,
		method:    meth,
		object:    obj,
		Signature: sig,
		Synthetic: description,
		Prog:      prog,
		pos:       obj.Pos(),
	}
	fn.startBody()
	fn.addSpilledParam(recv)
	createParams(fn, start)

	indices := meth.Index()

	var v Value = fn.Locals[0] // spilled receiver
	if isPointer(meth.Recv()) {
		v = emitLoad(fn, v)

		// For simple indirection wrappers, perform an informative nil-check:
		// "value method (T).f called using nil *T pointer"
		if len(indices) == 1 && !isPointer(recvType(obj)) {
			var c Call
			c.Call.Value = &Builtin{
				name: "ssa:wrapnilchk",
				sig: types.NewSignature(nil, nil,
					types.NewTuple(anonVar(meth.Recv()), anonVar(tString), anonVar(tString)),
					types.NewTuple(anonVar(meth.Recv())), false),
			}
			c.Call.Args = []Value{
				v,
				stringConst(deref(meth.Recv()).String()),
				stringConst(meth.Obj().Name()),
			}
			c.setType(v.Type())
			v = fn.emit(&c)
		}
	}

	// Invariant: v is a pointer, either
	//   value of *A receiver param, or
	// address of  A spilled receiver.

	// We use pointer arithmetic (FieldAddr possibly followed by
	// Load) in preference to value extraction (Field possibly
	// preceded by Load).

	v = emitImplicitSelections(fn, v, indices[:len(indices)-1])

	// Invariant: v is a pointer, either
	//   value of implicit *C field, or
	// address of implicit  C field.

	var c Call
	if r := recvType(obj); !isInterface(r) { // concrete method
		if !isPointer(r) {
			v = emitLoad(fn, v)
		}
		c.Call.Value = prog.declaredFunc(obj)
		c.Call.Args = append(c.Call.Args, v)
	} else {
		c.Call.Method = obj
		c.Call.Value = emitLoad(fn, v)
	}
	for _, arg := range fn.Params[1:] {
		c.Call.Args = append(c.Call.Args, arg)
	}
	emitTailCall(fn, &c)
	fn.finishBody()
	return fn
}

// createParams creates parameters for wrapper method fn based on its
// Signature.Params, which do not include the receiver.
// start is the index of the first regular parameter to use.
//
func createParams(fn *Function, start int) {
	var last *Parameter
	tparams := fn.Signature.Params()
	for i, n := start, tparams.Len(); i < n; i++ {
		last = fn.addParamObj(tparams.At(i))
	}
	if fn.Signature.Variadic() {
		last.typ = types.NewSlice(last.typ)
	}
}

// -- bounds -----------------------------------------------------------

// makeBound returns a bound method wrapper (or "bound"), a synthetic
// function that delegates to a concrete or interface method denoted
// by obj.  The resulting function has no receiver, but has one free
// variable which will be used as the method's receiver in the
// tail-call.
//
// Use MakeClosure with such a wrapper to construct a bound method
// closure.  e.g.:
//
//   type T int          or:  type T interface { meth() }
//   func (t T) meth()
//   var t T
//   f := t.meth
//   f() // calls t.meth()
//
// f is a closure of a synthetic wrapper defined as if by:
//
//   f := func() { return t.meth() }
//
// Unlike makeWrapper, makeBound need perform no indirection or field
// selections because that can be done before the closure is
// constructed.
//
// EXCLUSIVE_LOCKS_ACQUIRED(meth.Prog.methodsMu)
//
func makeBound(prog *Program, obj *types.Func) *Function {
	prog.methodsMu.Lock()
	defer prog.methodsMu.Unlock()
	fn, ok := prog.bounds[obj]
	if !ok {
		description := fmt.Sprintf("bound method wrapper for %s", obj)
		if prog.mode&LogSource != 0 {
			defer logStack("%s", description)()
		}
		fn = &Function{
			name:      obj.Name() + "$bound",
			object:    obj,
			Signature: changeRecv(obj.Type().(*types.Signature), nil), // drop receiver
			Synthetic: description,
			Prog:      prog,
			pos:       obj.Pos(),
		}

		fv := &FreeVar{name: "recv", typ: recvType(obj), parent: fn}
		fn.FreeVars = []*FreeVar{fv}
		fn.startBody()
		createParams(fn, 0)
		var c Call

		if !isInterface(recvType(obj)) { // concrete
			c.Call.Value = prog.declaredFunc(obj)
			c.Call.Args = []Value{fv}
		} else {
			c.Call.Value = fv
			c.Call.Method = obj
		}
		for _, arg := range fn.Params {
			c.Call.Args = append(c.Call.Args, arg)
		}
		emitTailCall(fn, &c)
		fn.finishBody()

		prog.bounds[obj] = fn
	}
	return fn
}

// -- thunks -----------------------------------------------------------

// makeThunk returns a thunk, a synthetic function that delegates to a
// concrete or interface method denoted by sel.Obj().  The resulting
// function has no receiver, but has an additional (first) regular
// parameter.
//
// Precondition: sel.Kind() == types.MethodExpr.
//
//   type T int          or:  type T interface { meth() }
//   func (t T) meth()
//   f := T.meth
//   var t T
//   f(t) // calls t.meth()
//
// f is a synthetic wrapper defined as if by:
//
//   f := func(t T) { return t.meth() }
//
// TODO(adonovan): opt: currently the stub is created even when used
// directly in a function call: C.f(i, 0).  This is less efficient
// than inlining the stub.
//
// EXCLUSIVE_LOCKS_ACQUIRED(meth.Prog.methodsMu)
//
func makeThunk(prog *Program, sel *types.Selection) *Function {
	if sel.Kind() != types.MethodExpr {
		panic(sel)
	}

	// TODO(adonovan): opt: canonicalize the recv Type to avoid
	// construct unnecessary duplicate thunks.
	key := selectionKey{
		kind:     sel.Kind(),
		recv:     sel.Recv(),
		obj:      sel.Obj(),
		index:    fmt.Sprint(sel.Index()),
		indirect: sel.Indirect(),
	}

	prog.methodsMu.Lock()
	defer prog.methodsMu.Unlock()
	fn, ok := prog.thunks[key]
	if !ok {
		fn = makeWrapper(prog, sel)
		if fn.Signature.Recv() != nil {
			panic(fn) // unexpected receiver
		}
		prog.thunks[key] = fn
	}
	return fn
}

func changeRecv(s *types.Signature, recv *types.Var) *types.Signature {
	return types.NewSignature(nil, recv, s.Params(), s.Results(), s.Variadic())
}

// recvType returns the receiver type of method obj.
func recvType(obj *types.Func) types.Type {
	return obj.Type().(*types.Signature).Recv().Type()
}

func isInterface(T types.Type) bool {
	_, ok := T.Underlying().(*types.Interface)
	return ok
}

// selectionKey is like types.Selection but a usable map key.
type selectionKey struct {
	kind     types.SelectionKind
	recv     types.Type
	obj      types.Object
	index    string
	indirect bool
}
