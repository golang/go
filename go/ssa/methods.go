// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// This file defines utilities for population of method sets.

import (
	"fmt"
	"go/types"

	"golang.org/x/tools/internal/typeparams"
)

// MethodValue returns the Function implementing method sel, building
// wrapper methods on demand.  It returns nil if sel denotes an
// abstract (interface or parameterized) method.
//
// Precondition: sel.Kind() == MethodVal.
//
// Thread-safe.
//
// EXCLUSIVE_LOCKS_ACQUIRED(prog.methodsMu)
func (prog *Program) MethodValue(sel *types.Selection) *Function {
	if sel.Kind() != types.MethodVal {
		panic(fmt.Sprintf("MethodValue(%s) kind != MethodVal", sel))
	}
	T := sel.Recv()
	if isInterface(T) {
		return nil // abstract method (interface)
	}
	if prog.mode&LogSource != 0 {
		defer logStack("MethodValue %s %v", T, sel)()
	}

	var m *Function
	b := builder{created: &creator{}}

	prog.methodsMu.Lock()
	// Checks whether a type param is reachable from T.
	// This is an expensive check. May need to be optimized later.
	if !prog.parameterized.isParameterized(T) {
		m = prog.addMethod(prog.createMethodSet(T), sel, b.created)
	}
	prog.methodsMu.Unlock()

	if m == nil {
		return nil // abstract method (generic)
	}
	for !b.done() {
		b.buildCreated()
		b.needsRuntimeTypes()
	}
	return m
}

// LookupMethod returns the implementation of the method of type T
// identified by (pkg, name).  It returns nil if the method exists but
// is abstract, and panics if T has no such method.
func (prog *Program) LookupMethod(T types.Type, pkg *types.Package, name string) *Function {
	sel := prog.MethodSets.MethodSet(T).Lookup(pkg, name)
	if sel == nil {
		panic(fmt.Sprintf("%s has no method %s", T, types.Id(pkg, name)))
	}
	return prog.MethodValue(sel)
}

// methodSet contains the (concrete) methods of a concrete type (non-interface, non-parameterized).
type methodSet struct {
	mapping  map[string]*Function // populated lazily
	complete bool                 // mapping contains all methods
}

// Precondition: T is a concrete type, e.g. !isInterface(T) and not parameterized.
// EXCLUSIVE_LOCKS_REQUIRED(prog.methodsMu)
func (prog *Program) createMethodSet(T types.Type) *methodSet {
	if prog.mode&SanityCheckFunctions != 0 {
		if isInterface(T) || prog.parameterized.isParameterized(T) {
			panic("type is interface or parameterized")
		}
	}
	mset, ok := prog.methodSets.At(T).(*methodSet)
	if !ok {
		mset = &methodSet{mapping: make(map[string]*Function)}
		prog.methodSets.Set(T, mset)
	}
	return mset
}

// Adds any created functions to cr.
// Precondition: T is a concrete type, e.g. !isInterface(T) and not parameterized.
// EXCLUSIVE_LOCKS_REQUIRED(prog.methodsMu)
func (prog *Program) addMethod(mset *methodSet, sel *types.Selection, cr *creator) *Function {
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
			fn = makeWrapper(prog, sel, cr)
		} else {
			fn = prog.originFunc(obj)
			if len(fn._TypeParams) > 0 { // instantiate
				targs := receiverTypeArgs(obj)
				fn = prog.instances[fn].lookupOrCreate(targs, cr)
			}
		}
		if fn.Signature.Recv() == nil {
			panic(fn) // missing receiver
		}
		mset.mapping[id] = fn
	}
	return fn
}

// RuntimeTypes returns a new unordered slice containing all
// concrete types in the program for which a complete (non-empty)
// method set is required at run-time.
//
// Thread-safe.
//
// EXCLUSIVE_LOCKS_ACQUIRED(prog.methodsMu)
func (prog *Program) RuntimeTypes() []types.Type {
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

// declaredFunc returns the concrete function/method denoted by obj.
// Panic ensues if there is none.
func (prog *Program) declaredFunc(obj *types.Func) *Function {
	if v := prog.packageLevelMember(obj); v != nil {
		return v.(*Function)
	}
	panic("no concrete method: " + obj.String())
}

// needMethodsOf ensures that runtime type information (including the
// complete method set) is available for the specified type T and all
// its subcomponents.
//
// needMethodsOf must be called for at least every type that is an
// operand of some MakeInterface instruction, and for the type of
// every exported package member.
//
// Adds any created functions to cr.
//
// Precondition: T is not a method signature (*Signature with Recv()!=nil).
// Precondition: T is not parameterized.
//
// Thread-safe.  (Called via Package.build from multiple builder goroutines.)
//
// TODO(adonovan): make this faster.  It accounts for 20% of SSA build time.
//
// EXCLUSIVE_LOCKS_ACQUIRED(prog.methodsMu)
func (prog *Program) needMethodsOf(T types.Type, cr *creator) {
	prog.methodsMu.Lock()
	prog.needMethods(T, false, cr)
	prog.methodsMu.Unlock()
}

// Precondition: T is not a method signature (*Signature with Recv()!=nil).
// Precondition: T is not parameterized.
// Recursive case: skip => don't create methods for T.
//
// EXCLUSIVE_LOCKS_REQUIRED(prog.methodsMu)
func (prog *Program) needMethods(T types.Type, skip bool, cr *creator) {
	// Each package maintains its own set of types it has visited.
	if prevSkip, ok := prog.runtimeTypes.At(T).(bool); ok {
		// needMethods(T) was previously called
		if !prevSkip || skip {
			return // already seen, with same or false 'skip' value
		}
	}
	prog.runtimeTypes.Set(T, skip)

	tmset := prog.MethodSets.MethodSet(T)

	if !skip && !isInterface(T) && tmset.Len() > 0 {
		// Create methods of T.
		mset := prog.createMethodSet(T)
		if !mset.complete {
			mset.complete = true
			n := tmset.Len()
			for i := 0; i < n; i++ {
				prog.addMethod(mset, tmset.At(i), cr)
			}
		}
	}

	// Recursion over signatures of each method.
	for i := 0; i < tmset.Len(); i++ {
		sig := tmset.At(i).Type().(*types.Signature)
		prog.needMethods(sig.Params(), false, cr)
		prog.needMethods(sig.Results(), false, cr)
	}

	switch t := T.(type) {
	case *types.Basic:
		// nop

	case *types.Interface:
		// nop---handled by recursion over method set.

	case *types.Pointer:
		prog.needMethods(t.Elem(), false, cr)

	case *types.Slice:
		prog.needMethods(t.Elem(), false, cr)

	case *types.Chan:
		prog.needMethods(t.Elem(), false, cr)

	case *types.Map:
		prog.needMethods(t.Key(), false, cr)
		prog.needMethods(t.Elem(), false, cr)

	case *types.Signature:
		if t.Recv() != nil {
			panic(fmt.Sprintf("Signature %s has Recv %s", t, t.Recv()))
		}
		prog.needMethods(t.Params(), false, cr)
		prog.needMethods(t.Results(), false, cr)

	case *types.Named:
		// A pointer-to-named type can be derived from a named
		// type via reflection.  It may have methods too.
		prog.needMethods(types.NewPointer(T), false, cr)

		// Consider 'type T struct{S}' where S has methods.
		// Reflection provides no way to get from T to struct{S},
		// only to S, so the method set of struct{S} is unwanted,
		// so set 'skip' flag during recursion.
		prog.needMethods(t.Underlying(), true, cr)

	case *types.Array:
		prog.needMethods(t.Elem(), false, cr)

	case *types.Struct:
		for i, n := 0, t.NumFields(); i < n; i++ {
			prog.needMethods(t.Field(i).Type(), false, cr)
		}

	case *types.Tuple:
		for i, n := 0, t.Len(); i < n; i++ {
			prog.needMethods(t.At(i).Type(), false, cr)
		}

	case *typeparams.TypeParam:
		panic(T) // type parameters are always abstract.

	case *typeparams.Union:
		// nop

	default:
		panic(T)
	}
}
