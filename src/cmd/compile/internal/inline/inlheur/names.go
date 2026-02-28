// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inlheur

import (
	"cmd/compile/internal/ir"
	"go/constant"
)

// nameFinder provides a set of "isXXX" query methods for clients to
// ask whether a given AST node corresponds to a function, a constant
// value, and so on. These methods use an underlying ir.ReassignOracle
// to return more precise results in cases where an "interesting"
// value is assigned to a singly-defined local temp. Example:
//
//	const q = 101
//	fq := func() int { return q }
//	copyOfConstant := q
//	copyOfFunc := f
//	interestingCall(copyOfConstant, copyOfFunc)
//
// A name finder query method invoked on the arguments being passed to
// "interestingCall" will be able detect that 'copyOfConstant' always
// evaluates to a constant (even though it is in fact a PAUTO local
// variable). A given nameFinder can also operate without using
// ir.ReassignOracle (in cases where it is not practical to look
// at the entire function); in such cases queries will still work
// for explicit constant values and functions.
type nameFinder struct {
	ro *ir.ReassignOracle
}

// newNameFinder returns a new nameFinder object with a reassignment
// oracle initialized based on the function fn, or if fn is nil,
// without an underlying ReassignOracle.
func newNameFinder(fn *ir.Func) *nameFinder {
	var ro *ir.ReassignOracle
	if fn != nil {
		ro = &ir.ReassignOracle{}
		ro.Init(fn)
	}
	return &nameFinder{ro: ro}
}

// funcName returns the *ir.Name for the func or method
// corresponding to node 'n', or nil if n can't be proven
// to contain a function value.
func (nf *nameFinder) funcName(n ir.Node) *ir.Name {
	sv := n
	if nf.ro != nil {
		sv = nf.ro.StaticValue(n)
	}
	if name := ir.StaticCalleeName(sv); name != nil {
		return name
	}
	return nil
}

// isAllocatedMem returns true if node n corresponds to a memory
// allocation expression (make, new, or equivalent).
func (nf *nameFinder) isAllocatedMem(n ir.Node) bool {
	sv := n
	if nf.ro != nil {
		sv = nf.ro.StaticValue(n)
	}
	switch sv.Op() {
	case ir.OMAKESLICE, ir.ONEW, ir.OPTRLIT, ir.OSLICELIT:
		return true
	}
	return false
}

// constValue returns the underlying constant.Value for an AST node n
// if n is itself a constant value/expr, or if n is a singly assigned
// local containing constant expr/value (or nil not constant).
func (nf *nameFinder) constValue(n ir.Node) constant.Value {
	sv := n
	if nf.ro != nil {
		sv = nf.ro.StaticValue(n)
	}
	if sv.Op() == ir.OLITERAL {
		return sv.Val()
	}
	return nil
}

// isNil returns whether n is nil (or singly
// assigned local containing nil).
func (nf *nameFinder) isNil(n ir.Node) bool {
	sv := n
	if nf.ro != nil {
		sv = nf.ro.StaticValue(n)
	}
	return sv.Op() == ir.ONIL
}

func (nf *nameFinder) staticValue(n ir.Node) ir.Node {
	if nf.ro == nil {
		return n
	}
	return nf.ro.StaticValue(n)
}

func (nf *nameFinder) reassigned(n *ir.Name) bool {
	if nf.ro == nil {
		return true
	}
	return nf.ro.Reassigned(n)
}

func (nf *nameFinder) isConcreteConvIface(n ir.Node) bool {
	sv := n
	if nf.ro != nil {
		sv = nf.ro.StaticValue(n)
	}
	if sv.Op() != ir.OCONVIFACE {
		return false
	}
	return !sv.(*ir.ConvExpr).X.Type().IsInterface()
}

func isSameFuncName(v1, v2 *ir.Name) bool {
	// NB: there are a few corner cases where pointer equality
	// doesn't work here, but this should be good enough for
	// our purposes here.
	return v1 == v2
}
