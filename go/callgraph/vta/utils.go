// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vta

import (
	"go/types"

	"golang.org/x/tools/go/callgraph"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/internal/typeparams"
)

func canAlias(n1, n2 node) bool {
	return isReferenceNode(n1) && isReferenceNode(n2)
}

func isReferenceNode(n node) bool {
	if _, ok := n.(nestedPtrInterface); ok {
		return true
	}
	if _, ok := n.(nestedPtrFunction); ok {
		return true
	}

	if _, ok := n.Type().(*types.Pointer); ok {
		return true
	}

	return false
}

// hasInFlow checks if a concrete type can flow to node `n`.
// Returns yes iff the type of `n` satisfies one the following:
//  1. is an interface
//  2. is a (nested) pointer to interface (needed for, say,
//     slice elements of nested pointers to interface type)
//  3. is a function type (needed for higher-order type flow)
//  4. is a (nested) pointer to function (needed for, say,
//     slice elements of nested pointers to function type)
//  5. is a global Recover or Panic node
func hasInFlow(n node) bool {
	if _, ok := n.(panicArg); ok {
		return true
	}
	if _, ok := n.(recoverReturn); ok {
		return true
	}

	t := n.Type()

	if i := interfaceUnderPtr(t); i != nil {
		return true
	}
	if f := functionUnderPtr(t); f != nil {
		return true
	}

	return types.IsInterface(t) || isFunction(t)
}

func isFunction(t types.Type) bool {
	_, ok := t.Underlying().(*types.Signature)
	return ok
}

// interfaceUnderPtr checks if type `t` is a potentially nested
// pointer to interface and if yes, returns the interface type.
// Otherwise, returns nil.
func interfaceUnderPtr(t types.Type) types.Type {
	seen := make(map[types.Type]bool)
	var visit func(types.Type) types.Type
	visit = func(t types.Type) types.Type {
		if seen[t] {
			return nil
		}
		seen[t] = true

		p, ok := t.Underlying().(*types.Pointer)
		if !ok {
			return nil
		}

		if types.IsInterface(p.Elem()) {
			return p.Elem()
		}

		return visit(p.Elem())
	}
	return visit(t)
}

// functionUnderPtr checks if type `t` is a potentially nested
// pointer to function type and if yes, returns the function type.
// Otherwise, returns nil.
func functionUnderPtr(t types.Type) types.Type {
	seen := make(map[types.Type]bool)
	var visit func(types.Type) types.Type
	visit = func(t types.Type) types.Type {
		if seen[t] {
			return nil
		}
		seen[t] = true

		p, ok := t.Underlying().(*types.Pointer)
		if !ok {
			return nil
		}

		if isFunction(p.Elem()) {
			return p.Elem()
		}

		return visit(p.Elem())
	}
	return visit(t)
}

// sliceArrayElem returns the element type of type `t` that is
// expected to be a (pointer to) array, slice or string, consistent with
// the ssa.Index and ssa.IndexAddr instructions. Panics otherwise.
func sliceArrayElem(t types.Type) types.Type {
	switch u := t.Underlying().(type) {
	case *types.Pointer:
		switch e := u.Elem().Underlying().(type) {
		case *types.Array:
			return e.Elem()
		case *types.Interface:
			return sliceArrayElem(e) // e is a type param with matching element types.
		default:
			panic(t)
		}
	case *types.Array:
		return u.Elem()
	case *types.Slice:
		return u.Elem()
	case *types.Basic:
		return types.Typ[types.Byte]
	case *types.Interface: // type param.
		terms, err := typeparams.InterfaceTermSet(u)
		if err != nil || len(terms) == 0 {
			panic(t)
		}
		return sliceArrayElem(terms[0].Type()) // Element types must match.
	default:
		panic(t)
	}
}

// siteCallees computes a set of callees for call site `c` given program `callgraph`.
func siteCallees(c ssa.CallInstruction, callgraph *callgraph.Graph) []*ssa.Function {
	var matches []*ssa.Function

	node := callgraph.Nodes[c.Parent()]
	if node == nil {
		return nil
	}

	for _, edge := range node.Out {
		if edge.Site == c {
			matches = append(matches, edge.Callee.Func)
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
