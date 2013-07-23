package ssa

// This file defines utilities for method-set computation including
// synthesis of wrapper methods.
//
// Wrappers include:
// - promotion wrappers for methods of embedded fields.
// - interface method wrappers for closures of I.f.
// - bound method wrappers, for uncalled obj.Method closures.
// - indirection wrappers, for calls to T-methods on a *T receiver.

// TODO(adonovan): rename to wrappers.go.

import (
	"fmt"
	"go/token"

	"code.google.com/p/go.tools/go/types"
)

// MethodSet returns the method set for type typ, building wrapper
// methods as needed for embedded field promotion, and indirection for
// *T receiver types, etc.
// A nil result indicates an empty set.
//
// Thread-safe.
//
func (prog *Program) MethodSet(typ types.Type) MethodSet {
	return prog.populateMethodSet(typ, "")
}

// populateMethodSet returns the method set for typ, ensuring that it
// contains at least key id.  If id is empty, the entire method set is
// populated.
//
func (prog *Program) populateMethodSet(typ types.Type, id string) MethodSet {
	tmset := typ.MethodSet()
	n := tmset.Len()
	if n == 0 {
		return nil
	}
	if _, ok := deref(typ).Underlying().(*types.Interface); ok {
		// TODO(gri): fix: go/types bug: pointer-to-interface
		// has no methods---yet go/types says it has!
		return nil
	}

	if prog.mode&LogSource != 0 {
		defer logStack("MethodSet %s id=%s", typ, id)()
	}

	prog.methodsMu.Lock()
	defer prog.methodsMu.Unlock()

	mset, _ := prog.methodSets.At(typ).(MethodSet)
	if mset == nil {
		mset = make(MethodSet)
		prog.methodSets.Set(typ, mset)
	}

	if len(mset) < n {
		if id != "" { // single method
			// tmset.Lookup() is no use to us with only an Id string.
			if mset[id] == nil {
				for i := 0; i < n; i++ {
					obj := tmset.At(i)
					if obj.Id() == id {
						mset[id] = makeMethod(prog, typ, obj)
						return mset
					}
				}
			}
		}

		// complete set
		for i := 0; i < n; i++ {
			obj := tmset.At(i)
			if id := obj.Id(); mset[id] == nil {
				mset[id] = makeMethod(prog, typ, obj)
			}
		}
	}

	return mset
}

// LookupMethod returns the method id of type typ, building wrapper
// methods on demand.  It returns nil if the typ has no such method.
//
// Thread-safe.
//
func (prog *Program) LookupMethod(typ types.Type, id string) *Function {
	return prog.populateMethodSet(typ, id)[id]
}

// makeMethod returns the concrete Function for the method obj,
// adapted if necessary so that its receiver type is typ.
//
// EXCLUSIVE_LOCKS_REQUIRED(prog.methodsMu)
//
func makeMethod(prog *Program, typ types.Type, obj *types.Method) *Function {
	// Promoted method accessed via implicit field selections?
	if len(obj.Index()) > 1 {
		return promotionWrapper(prog, typ, obj)
	}

	method := prog.concreteMethods[obj.Func]
	if method == nil {
		panic("no concrete method for " + obj.Func.String())
	}

	// Call to method on T from receiver of type *T?
	if !isPointer(method.Signature.Recv().Type()) && isPointer(typ) {
		method = indirectionWrapper(method)
	}

	return method
}

// promotionWrapper returns a synthetic wrapper Function that performs
// a sequence of implicit field selections then tailcalls a "promoted"
// method.  For example, given these decls:
//
//    type A struct {B}
//    type B struct {*C}
//    type C ...
//    func (*C) f()
//
// then promotionWrapper(typ=A, obj={Func:(*C).f, Indices=[B,C,f]})
// synthesize this wrapper method:
//
//    func (a A) f() { return a.B.C->f() }
//
// prog is the program to which the synthesized method will belong.
// typ is the receiver type of the wrapper method.  obj is the
// type-checker's object for the promoted method; its Func may be a
// concrete or an interface method.
//
// EXCLUSIVE_LOCKS_REQUIRED(prog.methodsMu)
//
func promotionWrapper(prog *Program, typ types.Type, obj *types.Method) *Function {
	old := obj.Func.Type().(*types.Signature)
	sig := types.NewSignature(types.NewVar(token.NoPos, nil, "recv", typ), old.Params(), old.Results(), old.IsVariadic())

	// TODO(adonovan): include implicit field path in description.
	description := fmt.Sprintf("promotion wrapper for (%s).%s", old.Recv(), obj.Func.Name())

	if prog.mode&LogSource != 0 {
		defer logStack("make %s to (%s)", description, typ)()
	}
	fn := &Function{
		name:      obj.Name(),
		object:    obj,
		Signature: sig,
		Synthetic: description,
		Prog:      prog,
		pos:       obj.Pos(),
	}
	fn.startBody()
	fn.addSpilledParam(sig.Recv())
	createParams(fn)

	var v Value = fn.Locals[0] // spilled receiver
	if isPointer(typ) {
		v = emitLoad(fn, v)
	}

	// Invariant: v is a pointer, either
	//   value of *A receiver param, or
	// address of  A spilled receiver.

	// We use pointer arithmetic (FieldAddr possibly followed by
	// Load) in preference to value extraction (Field possibly
	// preceded by Load).

	indices := obj.Index()
	v = emitImplicitSelections(fn, v, indices[:len(indices)-1])

	// Invariant: v is a pointer, either
	//   value of implicit *C field, or
	// address of implicit  C field.

	var c Call
	if _, ok := old.Recv().Type().Underlying().(*types.Interface); !ok { // concrete method
		if !isPointer(old.Recv().Type()) {
			v = emitLoad(fn, v)
		}
		m := prog.concreteMethods[obj.Func]
		if m == nil {
			panic("oops: " + fn.Synthetic)
		}
		c.Call.Func = m
		c.Call.Args = append(c.Call.Args, v)
	} else {
		c.Call.Method = indices[len(indices)-1]
		c.Call.Recv = emitLoad(fn, v)
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
//
func createParams(fn *Function) {
	var last *Parameter
	tparams := fn.Signature.Params()
	for i, n := 0, tparams.Len(); i < n; i++ {
		last = fn.addParamObj(tparams.At(i))
	}
	if fn.Signature.IsVariadic() {
		last.typ = types.NewSlice(last.typ)
	}
}

// Wrappers for standalone interface methods ----------------------------------

// interfaceMethodWrapper returns a synthetic wrapper function permitting a
// method id of interface typ to be called like a standalone function,
// e.g.:
//
//   type I interface { f(x int) R }
//   m := I.f  // wrapper
//   var i I
//   m(i, 0)
//
// The wrapper is defined as if by:
//
//   func I.f(i I, x int, ...) R {
//     return i.f(x, ...)
//   }
//
// TODO(adonovan): opt: currently the stub is created even when used
// in call position: I.f(i, 0).  Clearly this is suboptimal.
//
// EXCLUSIVE_LOCKS_ACQUIRED(prog.methodsMu)
//
func interfaceMethodWrapper(prog *Program, typ types.Type, id string) *Function {
	index, meth := interfaceMethodIndex(typ.Underlying().(*types.Interface), id)
	prog.methodsMu.Lock()
	defer prog.methodsMu.Unlock()
	// If one interface embeds another they'll share the same
	// wrappers for common methods.  This is safe, but it might
	// confuse some tools because of the implicit interface
	// conversion applied to the first argument.  If this becomes
	// a problem, we should include 'typ' in the memoization key.
	fn, ok := prog.ifaceMethodWrappers[meth]
	if !ok {
		if prog.mode&LogSource != 0 {
			defer logStack("interfaceMethodWrapper %s.%s", typ, id)()
		}
		fn = &Function{
			name:      meth.Name(),
			object:    meth,
			Signature: meth.Type().(*types.Signature),
			Synthetic: fmt.Sprintf("interface method wrapper for %s.%s", typ, id),
			pos:       meth.Pos(),
			Prog:      prog,
		}
		fn.startBody()
		fn.addParam("recv", typ, token.NoPos)
		createParams(fn)
		var c Call
		c.Call.Method = index
		c.Call.Recv = fn.Params[0]
		for _, arg := range fn.Params[1:] {
			c.Call.Args = append(c.Call.Args, arg)
		}
		emitTailCall(fn, &c)
		fn.finishBody()

		prog.ifaceMethodWrappers[meth] = fn
	}
	return fn
}

// Wrappers for bound methods -------------------------------------------------

// boundMethodWrapper returns a synthetic wrapper function that
// delegates to a concrete method.  The wrapper has one free variable,
// the method's receiver.  Use MakeClosure with such a wrapper to
// construct a bound-method closure.
// e.g.:
//
//   type T int
//   func (t T) meth()
//   var t T
//   f := t.meth
//   f() // calls t.meth()
//
// f is a closure of a synthetic wrapper defined as if by:
//
//   f := func() { return t.meth() }
//
// EXCLUSIVE_LOCKS_ACQUIRED(meth.Prog.methodsMu)
//
func boundMethodWrapper(meth *Function) *Function {
	prog := meth.Prog
	prog.methodsMu.Lock()
	defer prog.methodsMu.Unlock()
	fn, ok := prog.boundMethodWrappers[meth]
	if !ok {
		if prog.mode&LogSource != 0 {
			defer logStack("boundMethodWrapper %s", meth)()
		}
		s := meth.Signature
		fn = &Function{
			name:      "bound$" + meth.String(),
			Signature: types.NewSignature(nil, s.Params(), s.Results(), s.IsVariadic()), // drop recv
			Synthetic: "bound method wrapper for " + meth.String(),
			Prog:      prog,
			pos:       meth.Pos(),
		}

		cap := &Capture{name: "recv", typ: s.Recv().Type(), parent: fn}
		fn.FreeVars = []*Capture{cap}
		fn.startBody()
		createParams(fn)
		var c Call
		c.Call.Func = meth
		c.Call.Args = []Value{cap}
		for _, arg := range fn.Params {
			c.Call.Args = append(c.Call.Args, arg)
		}
		emitTailCall(fn, &c)
		fn.finishBody()

		prog.boundMethodWrappers[meth] = fn
	}
	return fn
}

// Receiver indirection wrapper ------------------------------------

// indirectionWrapper returns a synthetic method with *T receiver
// that delegates to meth, which has a T receiver.
//
//      func (recv *T) f(...) ... {
//              return (*recv).f(...)
//      }
//
// EXCLUSIVE_LOCKS_REQUIRED(meth.Prog.methodsMu)
//
func indirectionWrapper(meth *Function) *Function {
	prog := meth.Prog
	fn, ok := prog.indirectionWrappers[meth]
	if !ok {
		if prog.mode&LogSource != 0 {
			defer logStack("makeIndirectionWrapper %s", meth)()
		}

		s := meth.Signature
		recv := types.NewVar(token.NoPos, meth.Pkg.Object, "recv",
			types.NewPointer(s.Recv().Type()))
		// TODO(adonovan): is there a *types.Func for this method?
		fn = &Function{
			name:      meth.Name(),
			Signature: types.NewSignature(recv, s.Params(), s.Results(), s.IsVariadic()),
			Prog:      prog,
			Synthetic: "receiver indirection wrapper for " + meth.String(),
			pos:       meth.Pos(),
		}

		fn.startBody()
		fn.addParamObj(recv)
		createParams(fn)
		// TODO(adonovan): consider emitting a nil-pointer check here
		// with a nice error message, like gc does.
		var c Call
		c.Call.Func = meth
		c.Call.Args = append(c.Call.Args, emitLoad(fn, fn.Params[0]))
		for _, arg := range fn.Params[1:] {
			c.Call.Args = append(c.Call.Args, arg)
		}
		emitTailCall(fn, &c)
		fn.finishBody()

		prog.indirectionWrappers[meth] = fn
	}
	return fn
}
