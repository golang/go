// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssautil

// This file implements discovery of switch and type-switch constructs
// from low-level control flow.
//
// Many techniques exist for compiling a high-level switch with
// constant cases to efficient machine code.  The optimal choice will
// depend on the data type, the specific case values, the code in the
// body of each case, and the hardware.
// Some examples:
// - a lookup table (for a switch that maps constants to constants)
// - a computed goto
// - a binary tree
// - a perfect hash
// - a two-level switch (to partition constant strings by their first byte).

import (
	"bytes"
	"fmt"
	"go/token"

	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/ssa"
)

// A ConstCase represents a single constant comparison.
// It is part of a Switch.
type ConstCase struct {
	Block *ssa.BasicBlock // block performing the comparison
	Body  *ssa.BasicBlock // body of the case
	Value *ssa.Const      // case comparand
}

// A TypeCase represents a single type assertion.
// It is part of a Switch.
type TypeCase struct {
	Block   *ssa.BasicBlock // block performing the type assert
	Body    *ssa.BasicBlock // body of the case
	Type    types.Type      // case type
	Binding ssa.Value       // value bound by this case
}

// A Switch is a logical high-level control flow operation
// (a multiway branch) discovered by analysis of a CFG containing
// only if/else chains.  It is not part of the ssa.Instruction set.
//
// One of ConstCases and TypeCases has length >= 2;
// the other is nil.
//
// In a value switch, the list of cases may contain duplicate constants.
// A type switch may contain duplicate types, or types assignable
// to an interface type also in the list.
// TODO(adonovan): eliminate such duplicates.
//
type Switch struct {
	Start      *ssa.BasicBlock // block containing start of if/else chain
	X          ssa.Value       // the switch operand
	ConstCases []ConstCase     // ordered list of constant comparisons
	TypeCases  []TypeCase      // ordered list of type assertions
	Default    *ssa.BasicBlock // successor if all comparisons fail
}

func (sw *Switch) String() string {
	// We represent each block by the String() of its
	// first Instruction, e.g. "print(42:int)".
	var buf bytes.Buffer
	if sw.ConstCases != nil {
		fmt.Fprintf(&buf, "switch %s {\n", sw.X.Name())
		for _, c := range sw.ConstCases {
			fmt.Fprintf(&buf, "case %s: %s\n", c.Value, c.Body.Instrs[0])
		}
	} else {
		fmt.Fprintf(&buf, "switch %s.(type) {\n", sw.X.Name())
		for _, c := range sw.TypeCases {
			fmt.Fprintf(&buf, "case %s %s: %s\n",
				c.Binding.Name(), c.Type, c.Body.Instrs[0])
		}
	}
	if sw.Default != nil {
		fmt.Fprintf(&buf, "default: %s\n", sw.Default.Instrs[0])
	}
	fmt.Fprintf(&buf, "}")
	return buf.String()
}

// Switches examines the control-flow graph of fn and returns the
// set of inferred value and type switches.  A value switch tests an
// ssa.Value for equality against two or more compile-time constant
// values.  Switches involving link-time constants (addresses) are
// ignored.  A type switch type-asserts an ssa.Value against two or
// more types.
//
// The switches are returned in dominance order.
//
// The resulting switches do not necessarily correspond to uses of the
// 'switch' keyword in the source: for example, a single source-level
// switch statement with non-constant cases may result in zero, one or
// many Switches, one per plural sequence of constant cases.
// Switches may even be inferred from if/else- or goto-based control flow.
// (In general, the control flow constructs of the source program
// cannot be faithfully reproduced from the SSA representation.)
//
func Switches(fn *ssa.Function) []Switch {
	// Traverse the CFG in dominance order, so we don't
	// enter an if/else-chain in the middle.
	var switches []Switch
	seen := make(map[*ssa.BasicBlock]bool) // TODO(adonovan): opt: use ssa.blockSet
	for _, b := range fn.DomPreorder() {
		if x, k := isComparisonBlock(b); x != nil {
			// Block b starts a switch.
			sw := Switch{Start: b, X: x}
			valueSwitch(&sw, k, seen)
			if len(sw.ConstCases) > 1 {
				switches = append(switches, sw)
			}
		}

		if y, x, T := isTypeAssertBlock(b); y != nil {
			// Block b starts a type switch.
			sw := Switch{Start: b, X: x}
			typeSwitch(&sw, y, T, seen)
			if len(sw.TypeCases) > 1 {
				switches = append(switches, sw)
			}
		}
	}
	return switches
}

func valueSwitch(sw *Switch, k *ssa.Const, seen map[*ssa.BasicBlock]bool) {
	b := sw.Start
	x := sw.X
	for x == sw.X {
		if seen[b] {
			break
		}
		seen[b] = true

		sw.ConstCases = append(sw.ConstCases, ConstCase{
			Block: b,
			Body:  b.Succs[0],
			Value: k,
		})
		b = b.Succs[1]
		if len(b.Instrs) > 2 {
			// Block b contains not just 'if x == k',
			// so it may have side effects that
			// make it unsafe to elide.
			break
		}
		if len(b.Preds) != 1 {
			// Block b has multiple predecessors,
			// so it cannot be treated as a case.
			break
		}
		x, k = isComparisonBlock(b)
	}
	sw.Default = b
}

func typeSwitch(sw *Switch, y ssa.Value, T types.Type, seen map[*ssa.BasicBlock]bool) {
	b := sw.Start
	x := sw.X
	for x == sw.X {
		if seen[b] {
			break
		}
		seen[b] = true

		sw.TypeCases = append(sw.TypeCases, TypeCase{
			Block:   b,
			Body:    b.Succs[0],
			Type:    T,
			Binding: y,
		})
		b = b.Succs[1]
		if len(b.Instrs) > 4 {
			// Block b contains not just
			//  {TypeAssert; Extract #0; Extract #1; If}
			// so it may have side effects that
			// make it unsafe to elide.
			break
		}
		if len(b.Preds) != 1 {
			// Block b has multiple predecessors,
			// so it cannot be treated as a case.
			break
		}
		y, x, T = isTypeAssertBlock(b)
	}
	sw.Default = b
}

// isComparisonBlock returns the operands (v, k) if a block ends with
// a comparison v==k, where k is a compile-time constant.
//
func isComparisonBlock(b *ssa.BasicBlock) (v ssa.Value, k *ssa.Const) {
	if n := len(b.Instrs); n >= 2 {
		if i, ok := b.Instrs[n-1].(*ssa.If); ok {
			if binop, ok := i.Cond.(*ssa.BinOp); ok && binop.Block() == b && binop.Op == token.EQL {
				if k, ok := binop.Y.(*ssa.Const); ok {
					return binop.X, k
				}
				if k, ok := binop.X.(*ssa.Const); ok {
					return binop.Y, k
				}
			}
		}
	}
	return
}

// isTypeAssertBlock returns the operands (y, x, T) if a block ends with
// a type assertion "if y, ok := x.(T); ok {".
//
func isTypeAssertBlock(b *ssa.BasicBlock) (y, x ssa.Value, T types.Type) {
	if n := len(b.Instrs); n >= 4 {
		if i, ok := b.Instrs[n-1].(*ssa.If); ok {
			if ext1, ok := i.Cond.(*ssa.Extract); ok && ext1.Block() == b && ext1.Index == 1 {
				if ta, ok := ext1.Tuple.(*ssa.TypeAssert); ok && ta.Block() == b {
					// hack: relies upon instruction ordering.
					if ext0, ok := b.Instrs[n-3].(*ssa.Extract); ok {
						return ext0, ta.X, ta.AssertedType
					}
				}
			}
		}
	}
	return
}
