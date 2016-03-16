// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "fmt"

// Block represents a basic block in the control flow graph of a function.
type Block struct {
	// A unique identifier for the block. The system will attempt to allocate
	// these IDs densely, but no guarantees.
	ID ID

	// The kind of block this is.
	Kind BlockKind

	// Subsequent blocks, if any. The number and order depend on the block kind.
	// All successors must be distinct (to make phi values in successors unambiguous).
	Succs []*Block

	// Inverse of successors.
	// The order is significant to Phi nodes in the block.
	Preds []*Block
	// TODO: predecessors is a pain to maintain. Can we somehow order phi
	// arguments by block id and have this field computed explicitly when needed?

	// A value that determines how the block is exited. Its value depends on the kind
	// of the block. For instance, a BlockIf has a boolean control value and BlockExit
	// has a memory control value.
	Control *Value

	// Auxiliary info for the block. Its value depends on the Kind.
	Aux interface{}

	// The unordered set of Values that define the operation of this block.
	// The list must include the control value, if any. (TODO: need this last condition?)
	// After the scheduling pass, this list is ordered.
	Values []*Value

	// The containing function
	Func *Func

	// Line number for block's control operation
	Line int32

	// Likely direction for branches.
	// If BranchLikely, Succs[0] is the most likely branch taken.
	// If BranchUnlikely, Succs[1] is the most likely branch taken.
	// Ignored if len(Succs) < 2.
	// Fatal if not BranchUnknown and len(Succs) > 2.
	Likely BranchPrediction

	// After flagalloc, records whether flags are live at the end of the block.
	FlagsLiveAtEnd bool

	// Storage for Succs, Preds, and Values
	succstorage [2]*Block
	predstorage [4]*Block
	valstorage  [8]*Value
}

//     kind           control    successors
//   ------------------------------------------
//     Exit        return mem                []
//    Plain               nil            [next]
//       If   a boolean Value      [then, else]
//     Call               mem  [nopanic, panic]  (control opcode should be OpCall or OpStaticCall)
type BlockKind int32

// short form print
func (b *Block) String() string {
	return fmt.Sprintf("b%d", b.ID)
}

// long form print
func (b *Block) LongString() string {
	s := b.Kind.String()
	if b.Aux != nil {
		s += fmt.Sprintf(" %s", b.Aux)
	}
	if b.Control != nil {
		s += fmt.Sprintf(" %s", b.Control)
	}
	if len(b.Succs) > 0 {
		s += " ->"
		for _, c := range b.Succs {
			s += " " + c.String()
		}
	}
	switch b.Likely {
	case BranchUnlikely:
		s += " (unlikely)"
	case BranchLikely:
		s += " (likely)"
	}
	return s
}

func (b *Block) SetControl(v *Value) {
	if w := b.Control; w != nil {
		w.Uses--
	}
	b.Control = v
	if v != nil {
		v.Uses++
	}
}

// AddEdgeTo adds an edge from block b to block c. Used during building of the
// SSA graph; do not use on an already-completed SSA graph.
func (b *Block) AddEdgeTo(c *Block) {
	b.Succs = append(b.Succs, c)
	c.Preds = append(c.Preds, b)
}

func (b *Block) Logf(msg string, args ...interface{})           { b.Func.Logf(msg, args...) }
func (b *Block) Log() bool                                      { return b.Func.Log() }
func (b *Block) Fatalf(msg string, args ...interface{})         { b.Func.Fatalf(msg, args...) }
func (b *Block) Unimplementedf(msg string, args ...interface{}) { b.Func.Unimplementedf(msg, args...) }

type BranchPrediction int8

const (
	BranchUnlikely = BranchPrediction(-1)
	BranchUnknown  = BranchPrediction(0)
	BranchLikely   = BranchPrediction(+1)
)
