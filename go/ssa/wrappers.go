// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// This file defines synthesis of Functions that delegate to declared
// methods; they come in three kinds:
//
// (1) wrappers: methods that wrap declared methods, performing
//     implicit pointer indirections and embedded field selections.
//
// (2) thunks: funcs that wrap declared methods.  Like wrappers,
//     thunks perform indirections and field selections. The thunk's
//     first parameter is used as the receiver for the method call.
//
// (3) bounds: funcs that wrap declared methods.  The bound's sole
//     free variable, supplied by a closure, is used as the receiver
//     for the method call.  No indirections or field selections are
//     performed since they can be done before the call.

import (
	"fmt"

	"go/types"
)

// -- wrappers -----------------------------------------------------------

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
func makeWrapper(prog *Program, sel *types.Selection) *Function {
	obj := sel.Obj().(*types.Func)       // the declared function
	sig := sel.Type().(*types.Signature) // type of this wrapper

	var recv *types.Var // wrapper's receiver or thunk's params[0]
	name := obj.Name()
	var description string
	var start int // first regular param
	if sel.Kind() == types.MethodExpr {
		name += "$thunk"
		description = "thunk"
		recv = sig.Params().At(0)
		start = 1
	} else {
		description = "wrapper"
		recv = sig.Recv()
	}

	description = fmt.Sprintf("%s for %s", description, sel.Obj())
	if prog.mode&LogSource != 0 {
		defer logStack("make %s to (%s)", description, recv.Type())()
	}
	fn := &Function{
		name:      name,
		method:    sel,
		object:    obj,
		Signature: sig,
		Synthetic: description,
		Prog:      prog,
		pos:       obj.Pos(),
		info:      nil, // info is not set on wrappers.
	}
	fn.startBody()
	fn.addSpilledParam(recv)
	createParams(fn, start)

	indices := sel.Index()

	var v Value = fn.Locals[0] // spilled receiver
	if isPointer(sel.Recv()) {
		v = emitLoad(fn, v)

		// For simple indirection wrappers, perform an informative nil-check:
		// "value method (T).f called using nil *T pointer"
		if len(indices) == 1 && !isPointer(recvType(obj)) {
			var c Call
			c.Call.Value = &Builtin{
				name: "ssa:wrapnilchk",
				sig: types.NewSignature(nil,
					types.NewTuple(anonVar(sel.Recv()), anonVar(tString), anonVar(tString)),
					types.NewTuple(anonVar(sel.Recv())), false),
			}
			c.Call.Args = []Value{
				v,
				stringConst(deref(sel.Recv()).String()),
				stringConst(sel.Obj().Name()),
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
	tparams := fn.Signature.Params()
	for i, n := start, tparams.Len(); i < n; i++ {
		fn.addParamObj(tparams.At(i))
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
			info:      nil, // info is not set on wrappers.
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

	// Canonicalize sel.Recv() to avoid constructing duplicate thunks.
	canonRecv := prog.canon.Type(sel.Recv())
	key := selectionKey{
		kind:     sel.Kind(),
		recv:     canonRecv,
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
	return types.NewSignature(recv, s.Params(), s.Results(), s.Variadic())
}

// selectionKey is like types.Selection but a usable map key.
type selectionKey struct {
	kind     types.SelectionKind
	recv     types.Type // canonicalized via Program.canon
	obj      types.Object
	index    string
	indirect bool
}
