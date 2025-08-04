// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"cmd/compile/internal/base"
)

// A ReassignOracle efficiently answers queries about whether local
// variables are reassigned. This helper works by looking for function
// params and short variable declarations (e.g.
// https://go.dev/ref/spec#Short_variable_declarations) that are
// neither address taken nor subsequently re-assigned. It is intended
// to operate much like "ir.StaticValue" and "ir.Reassigned", but in a
// way that does just a single walk of the containing function (as
// opposed to a new walk on every call).
type ReassignOracle struct {
	fn *Func
	// maps candidate name to its defining assignment (or
	// for params, defining func).
	singleDef map[*Name]Node
}

// Init initializes the oracle based on the IR in function fn, laying
// the groundwork for future calls to the StaticValue and Reassigned
// methods. If the fn's IR is subsequently modified, Init must be
// called again.
func (ro *ReassignOracle) Init(fn *Func) {
	ro.fn = fn

	// Collect candidate map. Start by adding function parameters
	// explicitly.
	ro.singleDef = make(map[*Name]Node)
	sig := fn.Type()
	numParams := sig.NumRecvs() + sig.NumParams()
	for _, param := range fn.Dcl[:numParams] {
		if IsBlank(param) {
			continue
		}
		// For params, use func itself as defining node.
		ro.singleDef[param] = fn
	}

	// Walk the function body to discover any locals assigned
	// via ":=" syntax (e.g. "a := <expr>").
	var findLocals func(n Node) bool
	findLocals = func(n Node) bool {
		if nn, ok := n.(*Name); ok {
			if nn.Defn != nil && !nn.Addrtaken() && nn.Class == PAUTO {
				ro.singleDef[nn] = nn.Defn
			}
		} else if nn, ok := n.(*ClosureExpr); ok {
			Any(nn.Func, findLocals)
		}
		return false
	}
	Any(fn, findLocals)

	outerName := func(x Node) *Name {
		if x == nil {
			return nil
		}
		n, ok := OuterValue(x).(*Name)
		if ok {
			return n.Canonical()
		}
		return nil
	}

	// pruneIfNeeded examines node nn appearing on the left hand side
	// of assignment statement asn to see if it contains a reassignment
	// to any nodes in our candidate map ro.singleDef; if a reassignment
	// is found, the corresponding name is deleted from singleDef.
	pruneIfNeeded := func(nn Node, asn Node) {
		oname := outerName(nn)
		if oname == nil {
			return
		}
		defn, ok := ro.singleDef[oname]
		if !ok {
			return
		}
		// any assignment to a param invalidates the entry.
		paramAssigned := oname.Class == PPARAM
		// assignment to local ok iff assignment is its orig def.
		localAssigned := (oname.Class == PAUTO && asn != defn)
		if paramAssigned || localAssigned {
			// We found an assignment to name N that doesn't
			// correspond to its original definition; remove
			// from candidates.
			delete(ro.singleDef, oname)
		}
	}

	// Prune away anything that looks assigned. This code modeled after
	// similar code in ir.Reassigned; any changes there should be made
	// here as well.
	var do func(n Node) bool
	do = func(n Node) bool {
		switch n.Op() {
		case OAS:
			asn := n.(*AssignStmt)
			pruneIfNeeded(asn.X, n)
		case OAS2, OAS2FUNC, OAS2MAPR, OAS2DOTTYPE, OAS2RECV, OSELRECV2:
			asn := n.(*AssignListStmt)
			for _, p := range asn.Lhs {
				pruneIfNeeded(p, n)
			}
		case OASOP:
			asn := n.(*AssignOpStmt)
			pruneIfNeeded(asn.X, n)
		case ORANGE:
			rs := n.(*RangeStmt)
			pruneIfNeeded(rs.Key, n)
			pruneIfNeeded(rs.Value, n)
		case OCLOSURE:
			n := n.(*ClosureExpr)
			Any(n.Func, do)
		}
		return false
	}
	Any(fn, do)
}

// StaticValue method has the same semantics as the ir package function
// of the same name; see comments on [StaticValue].
func (ro *ReassignOracle) StaticValue(n Node) Node {
	arg := n
	for {
		if n.Op() == OCONVNOP {
			n = n.(*ConvExpr).X
			continue
		}

		if n.Op() == OINLCALL {
			n = n.(*InlinedCallExpr).SingleResult()
			continue
		}

		n1 := ro.staticValue1(n)
		if n1 == nil {
			if consistencyCheckEnabled {
				checkStaticValueResult(arg, n)
			}
			return n
		}
		n = n1
	}
}

func (ro *ReassignOracle) staticValue1(nn Node) Node {
	if nn.Op() != ONAME {
		return nil
	}
	n := nn.(*Name).Canonical()
	if n.Class != PAUTO {
		return nil
	}

	defn := n.Defn
	if defn == nil {
		return nil
	}

	var rhs Node
FindRHS:
	switch defn.Op() {
	case OAS:
		defn := defn.(*AssignStmt)
		rhs = defn.Y
	case OAS2:
		defn := defn.(*AssignListStmt)
		for i, lhs := range defn.Lhs {
			if lhs == n {
				rhs = defn.Rhs[i]
				break FindRHS
			}
		}
		base.FatalfAt(defn.Pos(), "%v missing from LHS of %v", n, defn)
	default:
		return nil
	}
	if rhs == nil {
		base.FatalfAt(defn.Pos(), "RHS is nil: %v", defn)
	}

	if _, ok := ro.singleDef[n]; !ok {
		return nil
	}

	return rhs
}

// Reassigned method has the same semantics as the ir package function
// of the same name; see comments on [Reassigned] for more info.
func (ro *ReassignOracle) Reassigned(n *Name) bool {
	_, ok := ro.singleDef[n]
	result := !ok
	if consistencyCheckEnabled {
		checkReassignedResult(n, result)
	}
	return result
}
