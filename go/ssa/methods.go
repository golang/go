// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// This file defines utilities for population of method sets.

import (
	"fmt"

	"golang.org/x/tools/go/types"
)

// Method returns the Function implementing method sel, building
// wrapper methods on demand.  It returns nil if sel denotes an
// abstract (interface) method.
//
// Precondition: sel.Kind() == MethodVal.
//
// TODO(adonovan): rename this to MethodValue because of the
// precondition, and for consistency with functions in source.go.
//
// Thread-safe.
//
// EXCLUSIVE_LOCKS_ACQUIRED(prog.methodsMu)
//
func (prog *Program) Method(sel *types.Selection) *Function {
	if sel.Kind() != types.MethodVal {
		panic(fmt.Sprintf("Method(%s) kind != MethodVal", sel))
	}
	T := sel.Recv()
	if isInterface(T) {
		return nil // abstract method
	}
	if prog.mode&LogSource != 0 {
		defer logStack("Method %s %v", T, sel)()
	}

	prog.methodsMu.Lock()
	defer prog.methodsMu.Unlock()

	return prog.addMethod(prog.createMethodSet(T), sel)
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
	// pkg.methodsMu not required; concurrent (build) phase is over.
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
