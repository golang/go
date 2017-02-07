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
//
package cha // import "golang.org/x/tools/go/callgraph/cha"

import (
	"go/types"

	"golang.org/x/tools/go/callgraph"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
	"golang.org/x/tools/go/types/typeutil"
)

// CallGraph computes the call graph of the specified program using the
// Class Hierarchy Analysis algorithm.
//
func CallGraph(prog *ssa.Program) *callgraph.Graph {
	cg := callgraph.New(nil) // TODO(adonovan) eliminate concept of rooted callgraph

	allFuncs := ssautil.AllFunctions(prog)

	// funcsBySig contains all functions, keyed by signature.  It is
	// the effective set of address-taken functions used to resolve
	// a dynamic call of a particular signature.
	var funcsBySig typeutil.Map // value is []*ssa.Function

	// methodsByName contains all methods,
	// grouped by name for efficient lookup.
	methodsByName := make(map[string][]*ssa.Function)

	// methodsMemo records, for every abstract method call call I.f on
	// interface type I, the set of concrete methods C.f of all
	// types C that satisfy interface I.
	methodsMemo := make(map[*types.Func][]*ssa.Function)
	lookupMethods := func(m *types.Func) []*ssa.Function {
		methods, ok := methodsMemo[m]
		if !ok {
			I := m.Type().(*types.Signature).Recv().Type().Underlying().(*types.Interface)
			for _, f := range methodsByName[m.Name()] {
				C := f.Signature.Recv().Type() // named or *named
				if types.Implements(C, I) {
					methods = append(methods, f)
				}
			}
			methodsMemo[m] = methods
		}
		return methods
	}

	for f := range allFuncs {
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
		// TODO(adonovan): opt: consider factoring the callgraph
		// API so that the Callers component of each edge is a
		// slice of nodes, not a singleton.
		for _, g := range callees {
			addEdge(fnode, site, g)
		}
	}

	for f := range allFuncs {
		fnode := cg.CreateNode(f)
		for _, b := range f.Blocks {
			for _, instr := range b.Instrs {
				if site, ok := instr.(ssa.CallInstruction); ok {
					call := site.Common()
					if call.IsInvoke() {
						addEdges(fnode, site, lookupMethods(call.Method))
					} else if g := call.StaticCallee(); g != nil {
						addEdge(fnode, site, g)
					} else if _, ok := call.Value.(*ssa.Builtin); !ok {
						callees, _ := funcsBySig.At(call.Signature()).([]*ssa.Function)
						addEdges(fnode, site, callees)
					}
				}
			}
		}
	}

	return cg
}
