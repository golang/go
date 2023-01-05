// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cha computes the call graph of a Go program using the Class
// Hierarchy Analysis (CHA) algorithm.
//
// CHA was first described in "Optimization of Object-Oriented Programs
// Using Static Class Hierarchy Analysis", Jeffrey Dean, David Grove,
// and Craig Chambers, ECOOP'95.
//
// CHA is related to RTA (see go/callgraph/rta); the difference is that
// CHA conservatively computes the entire "implements" relation between
// interfaces and concrete types ahead of time, whereas RTA uses dynamic
// programming to construct it on the fly as it encounters new functions
// reachable from main.  CHA may thus include spurious call edges for
// types that haven't been instantiated yet, or types that are never
// instantiated.
//
// Since CHA conservatively assumes that all functions are address-taken
// and all concrete types are put into interfaces, it is sound to run on
// partial programs, such as libraries without a main or test function.
package cha // import "golang.org/x/tools/go/callgraph/cha"

// TODO(zpavlinovic): update CHA for how it handles generic function bodies.

import (
	"go/types"

	"golang.org/x/tools/go/callgraph"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
	"golang.org/x/tools/go/types/typeutil"
)

// CallGraph computes the call graph of the specified program using the
// Class Hierarchy Analysis algorithm.
func CallGraph(prog *ssa.Program) *callgraph.Graph {
	cg := callgraph.New(nil) // TODO(adonovan) eliminate concept of rooted callgraph

	allFuncs := ssautil.AllFunctions(prog)

	calleesOf := lazyCallees(allFuncs)

	addEdge := func(fnode *callgraph.Node, site ssa.CallInstruction, g *ssa.Function) {
		gnode := cg.CreateNode(g)
		callgraph.AddEdge(fnode, site, gnode)
	}

	addEdges := func(fnode *callgraph.Node, site ssa.CallInstruction, callees []*ssa.Function) {
		// Because every call to a highly polymorphic and
		// frequently used abstract method such as
		// (io.Writer).Write is assumed to call every concrete
		// Write method in the program, the call graph can
		// contain a lot of duplication.
		//
		// TODO(taking): opt: consider making lazyCallees public.
		// Using the same benchmarks as callgraph_test.go, removing just
		// the explicit callgraph.Graph construction is 4x less memory
		// and is 37% faster.
		// CHA			86 ms/op	16 MB/op
		// lazyCallees	63 ms/op	 4 MB/op
		for _, g := range callees {
			addEdge(fnode, site, g)
		}
	}

	for f := range allFuncs {
		fnode := cg.CreateNode(f)
		for _, b := range f.Blocks {
			for _, instr := range b.Instrs {
				if site, ok := instr.(ssa.CallInstruction); ok {
					if g := site.Common().StaticCallee(); g != nil {
						addEdge(fnode, site, g)
					} else {
						addEdges(fnode, site, calleesOf(site))
					}
				}
			}
		}
	}

	return cg
}

// lazyCallees returns a function that maps a call site (in a function in fns)
// to its callees within fns.
//
// The resulting function is not concurrency safe.
func lazyCallees(fns map[*ssa.Function]bool) func(site ssa.CallInstruction) []*ssa.Function {
	// funcsBySig contains all functions, keyed by signature.  It is
	// the effective set of address-taken functions used to resolve
	// a dynamic call of a particular signature.
	var funcsBySig typeutil.Map // value is []*ssa.Function

	// methodsByName contains all methods,
	// grouped by name for efficient lookup.
	// (methodsById would be better but not every SSA method has a go/types ID.)
	methodsByName := make(map[string][]*ssa.Function)

	// An imethod represents an interface method I.m.
	// (There's no go/types object for it;
	// a *types.Func may be shared by many interfaces due to interface embedding.)
	type imethod struct {
		I  *types.Interface
		id string
	}
	// methodsMemo records, for every abstract method call I.m on
	// interface type I, the set of concrete methods C.m of all
	// types C that satisfy interface I.
	//
	// Abstract methods may be shared by several interfaces,
	// hence we must pass I explicitly, not guess from m.
	//
	// methodsMemo is just a cache, so it needn't be a typeutil.Map.
	methodsMemo := make(map[imethod][]*ssa.Function)
	lookupMethods := func(I *types.Interface, m *types.Func) []*ssa.Function {
		id := m.Id()
		methods, ok := methodsMemo[imethod{I, id}]
		if !ok {
			for _, f := range methodsByName[m.Name()] {
				C := f.Signature.Recv().Type() // named or *named
				if types.Implements(C, I) {
					methods = append(methods, f)
				}
			}
			methodsMemo[imethod{I, id}] = methods
		}
		return methods
	}

	for f := range fns {
		if f.Signature.Recv() == nil {
			// Package initializers can never be address-taken.
			if f.Name() == "init" && f.Synthetic == "package initializer" {
				continue
			}
			funcs, _ := funcsBySig.At(f.Signature).([]*ssa.Function)
			funcs = append(funcs, f)
			funcsBySig.Set(f.Signature, funcs)
		} else {
			methodsByName[f.Name()] = append(methodsByName[f.Name()], f)
		}
	}

	return func(site ssa.CallInstruction) []*ssa.Function {
		call := site.Common()
		if call.IsInvoke() {
			tiface := call.Value.Type().Underlying().(*types.Interface)
			return lookupMethods(tiface, call.Method)
		} else if g := call.StaticCallee(); g != nil {
			return []*ssa.Function{g}
		} else if _, ok := call.Value.(*ssa.Builtin); !ok {
			fns, _ := funcsBySig.At(call.Signature()).([]*ssa.Function)
			return fns
		}
		return nil
	}
}
