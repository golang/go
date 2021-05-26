// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vta

import (
	"go/types"

	"golang.org/x/tools/go/callgraph"
	"golang.org/x/tools/go/ssa"
)

func canAlias(n1, n2 node) bool {
	return isReferenceNode(n1) && isReferenceNode(n2)
}

func isReferenceNode(n node) bool {
	if _, ok := n.(nestedPtrInterface); ok {
		return true
	}

	if _, ok := n.Type().(*types.Pointer); ok {
		return true
	}

	return false
}

// hasInFlow checks if a concrete type can flow to node `n`.
// Returns yes iff the type of `n` satisfies one the following:
//  1) is an interface
//  2) is a (nested) pointer to interface (needed for, say,
//     slice elements of nested pointers to interface type)
//  3) is a function type (needed for higher-order type flow)
//  4) is a global Recover or Panic node
func hasInFlow(n node) bool {
	if _, ok := n.(panicArg); ok {
		return true
	}
	if _, ok := n.(recoverReturn); ok {
		return true
	}

	t := n.Type()

	if _, ok := t.Underlying().(*types.Signature); ok {
		return true
	}

	if i := interfaceUnderPtr(t); i != nil {
		return true
	}

	return isInterface(t)
}

// hasInitialTypes check if a node can have initial types.
// Returns true iff `n` is not a panic or recover node as
// those are artifical.
func hasInitialTypes(n node) bool {
	switch n.(type) {
	case panicArg, recoverReturn:
		return false
	default:
		return true
	}
}

func isInterface(t types.Type) bool {
	_, ok := t.Underlying().(*types.Interface)
	return ok
}

// interfaceUnderPtr checks if type `t` is a potentially nested
// pointer to interface and if yes, returns the interface type.
// Otherwise, returns nil.
func interfaceUnderPtr(t types.Type) types.Type {
	p, ok := t.Underlying().(*types.Pointer)
	if !ok {
		return nil
	}

	if isInterface(p.Elem()) {
		return p.Elem()
	}

	return interfaceUnderPtr(p.Elem())
}

// sliceArrayElem returns the element type of type `t` that is
// expected to be a (pointer to) array or slice, consistent with
// the ssa.Index and ssa.IndexAddr instructions. Panics otherwise.
func sliceArrayElem(t types.Type) types.Type {
	u := t.Underlying()

	if p, ok := u.(*types.Pointer); ok {
		u = p.Elem().Underlying()
	}

	if a, ok := u.(*types.Array); ok {
		return a.Elem()
	}
	return u.(*types.Slice).Elem()
}

// siteCallees computes a set of callees for call site `c` given program `callgraph`.
func siteCallees(c ssa.CallInstruction, callgraph *callgraph.Graph) []*ssa.Function {
	var matches []*ssa.Function

	node := callgraph.Nodes[c.Parent()]
	if node == nil {
		return nil
	}

	for _, edge := range node.Out {
		callee := edge.Callee.Func
		// Skip synthetic functions wrapped around source functions.
		if edge.Site == c && callee.Synthetic == "" {
			matches = append(matches, callee)
		}
	}
	return matches
}

func canHaveMethods(t types.Type) bool {
	if _, ok := t.(*types.Named); ok {
		return true
	}

	u := t.Underlying()
	switch u.(type) {
	case *types.Interface, *types.Signature, *types.Struct:
		return true
	default:
		return false
	}
}

// calls returns the set of call instructions in `f`.
func calls(f *ssa.Function) []ssa.CallInstruction {
	var calls []ssa.CallInstruction
	for _, bl := range f.Blocks {
		for _, instr := range bl.Instrs {
			if c, ok := instr.(ssa.CallInstruction); ok {
				calls = append(calls, c)
			}
		}
	}
	return calls
}

// intersect produces an intersection of functions in `fs1` and `fs2`.
func intersect(fs1, fs2 []*ssa.Function) []*ssa.Function {
	m := make(map[*ssa.Function]bool)
	for _, f := range fs1 {
		m[f] = true
	}

	var res []*ssa.Function
	for _, f := range fs2 {
		if m[f] {
			res = append(res, f)
		}
	}
	return res
}
