// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"go/token"
	"go/types"

	"golang.org/x/tools/cmd/guru/serial"
	"golang.org/x/tools/go/callgraph"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
)

// Callers reports the possible callers of the function
// immediately enclosing the specified source location.
//
func callers(q *Query) error {
	lconf := loader.Config{Build: q.Build}

	if err := setPTAScope(&lconf, q.Scope); err != nil {
		return err
	}

	// Load/parse/type-check the program.
	lprog, err := lconf.Load()
	if err != nil {
		return err
	}

	qpos, err := parseQueryPos(lprog, q.Pos, false)
	if err != nil {
		return err
	}

	prog := ssautil.CreateProgram(lprog, 0)

	ptaConfig, err := setupPTA(prog, lprog, q.PTALog, q.Reflection)
	if err != nil {
		return err
	}

	pkg := prog.Package(qpos.info.Pkg)
	if pkg == nil {
		return fmt.Errorf("no SSA package")
	}
	if !ssa.HasEnclosingFunction(pkg, qpos.path) {
		return fmt.Errorf("this position is not inside a function")
	}

	// Defer SSA construction till after errors are reported.
	prog.Build()

	target := ssa.EnclosingFunction(pkg, qpos.path)
	if target == nil {
		return fmt.Errorf("no SSA function built for this location (dead code?)")
	}

	// If the function is never address-taken, all calls are direct
	// and can be found quickly by inspecting the whole SSA program.
	cg := directCallsTo(target, entryPoints(ptaConfig.Mains))
	if cg == nil {
		// Run the pointer analysis, recording each
		// call found to originate from target.
		// (Pointer analysis may return fewer results than
		// directCallsTo because it ignores dead code.)
		ptaConfig.BuildCallGraph = true
		cg = ptrAnalysis(ptaConfig).CallGraph
	}
	cg.DeleteSyntheticNodes()
	edges := cg.CreateNode(target).In

	// TODO(adonovan): sort + dedup calls to ensure test determinism.

	q.Output(lprog.Fset, &callersResult{
		target:    target,
		callgraph: cg,
		edges:     edges,
	})
	return nil
}

// directCallsTo inspects the whole program and returns a callgraph
// containing edges for all direct calls to the target function.
// directCallsTo returns nil if the function is ever address-taken.
func directCallsTo(target *ssa.Function, entrypoints []*ssa.Function) *callgraph.Graph {
	cg := callgraph.New(nil) // use nil as root *Function
	targetNode := cg.CreateNode(target)

	// Is the function a program entry point?
	// If so, add edge from callgraph root.
	for _, f := range entrypoints {
		if f == target {
			callgraph.AddEdge(cg.Root, nil, targetNode)
		}
	}

	// Find receiver type (for methods).
	var recvType types.Type
	if recv := target.Signature.Recv(); recv != nil {
		recvType = recv.Type()
	}

	// Find all direct calls to function,
	// or a place where its address is taken.
	var space [32]*ssa.Value // preallocate
	for fn := range ssautil.AllFunctions(target.Prog) {
		for _, b := range fn.Blocks {
			for _, instr := range b.Instrs {
				// Is this a method (T).f of a concrete type T
				// whose runtime type descriptor is address-taken?
				// (To be fully sound, we would have to check that
				// the type doesn't make it to reflection as a
				// subelement of some other address-taken type.)
				if recvType != nil {
					if mi, ok := instr.(*ssa.MakeInterface); ok {
						if types.Identical(mi.X.Type(), recvType) {
							return nil // T is address-taken
						}
						if ptr, ok := mi.X.Type().(*types.Pointer); ok &&
							types.Identical(ptr.Elem(), recvType) {
							return nil // *T is address-taken
						}
					}
				}

				// Direct call to target?
				rands := instr.Operands(space[:0])
				if site, ok := instr.(ssa.CallInstruction); ok &&
					site.Common().Value == target {
					callgraph.AddEdge(cg.CreateNode(fn), site, targetNode)
					rands = rands[1:] // skip .Value (rands[0])
				}

				// Address-taken?
				for _, rand := range rands {
					if rand != nil && *rand == target {
						return nil
					}
				}
			}
		}
	}

	return cg
}

func entryPoints(mains []*ssa.Package) []*ssa.Function {
	var entrypoints []*ssa.Function
	for _, pkg := range mains {
		entrypoints = append(entrypoints, pkg.Func("init"))
		if main := pkg.Func("main"); main != nil && pkg.Pkg.Name() == "main" {
			entrypoints = append(entrypoints, main)
		}
	}
	return entrypoints
}

type callersResult struct {
	target    *ssa.Function
	callgraph *callgraph.Graph
	edges     []*callgraph.Edge
}

func (r *callersResult) PrintPlain(printf printfFunc) {
	root := r.callgraph.Root
	if r.edges == nil {
		printf(r.target, "%s is not reachable in this program.", r.target)
	} else {
		printf(r.target, "%s is called from these %d sites:", r.target, len(r.edges))
		for _, edge := range r.edges {
			if edge.Caller == root {
				printf(r.target, "the root of the call graph")
			} else {
				printf(edge, "\t%s from %s", edge.Description(), edge.Caller.Func)
			}
		}
	}
}

func (r *callersResult) JSON(fset *token.FileSet) []byte {
	var callers []serial.Caller
	for _, edge := range r.edges {
		callers = append(callers, serial.Caller{
			Caller: edge.Caller.Func.String(),
			Pos:    fset.Position(edge.Pos()).String(),
			Desc:   edge.Description(),
		})
	}
	return toJSON(callers)
}
