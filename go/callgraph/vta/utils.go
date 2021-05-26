// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vta

import (
	"go/types"
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
