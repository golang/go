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

	// Line number for block's control operation
	Line int32

	// The kind of block this is.
	Kind BlockKind

	// Likely direction for branches.
	// If BranchLikely, Succs[0] is the most likely branch taken.
	// If BranchUnlikely, Succs[1] is the most likely branch taken.
	// Ignored if len(Succs) < 2.
	// Fatal if not BranchUnknown and len(Succs) > 2.
	Likely BranchPrediction

	// After flagalloc, records whether flags are live at the end of the block.
	FlagsLiveAtEnd bool

	// Subsequent blocks, if any. The number and order depend on the block kind.
	Succs []Edge

	// Inverse of successors.
	// The order is significant to Phi nodes in the block.
	// TODO: predecessors is a pain to maintain. Can we somehow order phi
	// arguments by block id and have this field computed explicitly when needed?
	Preds []Edge

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

	// Storage for Succs, Preds, and Values
	succstorage [2]Edge
	predstorage [4]Edge
	valstorage  [9]*Value
}

// Edge represents a CFG edge.
// Example edges for b branching to either c or d.
// (c and d have other predecessors.)
//   b.Succs = [{c,3}, {d,1}]
//   c.Preds = [?, ?, ?, {b,0}]
//   d.Preds = [?, {b,1}, ?]
// These indexes allow us to edit the CFG in constant time.
// In addition, it informs phi ops in degenerate cases like:
// b:
//    if k then c else c
// c:
//    v = Phi(x, y)
// Then the indexes tell you whether x is chosen from
// the if or else branch from b.
//   b.Succs = [{c,0},{c,1}]
//   c.Preds = [{b,0},{b,1}]
// means x is chosen if k is true.
type Edge struct {
	// block edge goes to (in a Succs list) or from (in a Preds list)
	b *Block
	// index of reverse edge.  Invariant:
	//   e := x.Succs[idx]
	//   e.b.Preds[e.i] = Edge{x,idx}
	// and similarly for predecessors.
	i int
}

func (e Edge) Block() *Block {
	return e.b
}

//     kind           control    successors
//   ------------------------------------------
//     Exit        return mem                []
//    Plain               nil            [next]
//       If   a boolean Value      [then, else]
//     Call               mem  [nopanic, panic]  (control opcode should be OpCall or OpStaticCall)
type BlockKind int8

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
			s += " " + c.b.String()
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
	i := len(b.Succs)
	j := len(c.Preds)
	b.Succs = append(b.Succs, Edge{c, j})
	c.Preds = append(c.Preds, Edge{b, i})
}

// removePred removes the ith input edge from b.
// It is the responsibility of the caller to remove
// the corresponding successor edge.
func (b *Block) removePred(i int) {
	n := len(b.Preds) - 1
	if i != n {
		e := b.Preds[n]
		b.Preds[i] = e
		// Update the other end of the edge we moved.
		e.b.Succs[e.i].i = i
	}
	b.Preds[n] = Edge{}
	b.Preds = b.Preds[:n]
}

// removeSucc removes the ith output edge from b.
// It is the responsibility of the caller to remove
// the corresponding predecessor edge.
func (b *Block) removeSucc(i int) {
	n := len(b.Succs) - 1
	if i != n {
		e := b.Succs[n]
		b.Succs[i] = e
		// Update the other end of the edge we moved.
		e.b.Preds[e.i].i = i
	}
	b.Succs[n] = Edge{}
	b.Succs = b.Succs[:n]
}

func (b *Block) swapSuccessors() {
	if len(b.Succs) != 2 {
		b.Fatalf("swapSuccessors with len(Succs)=%d", len(b.Succs))
	}
	e0 := b.Succs[0]
	e1 := b.Succs[1]
	b.Succs[0] = e1
	b.Succs[1] = e0
	e0.b.Preds[e0.i].i = 1
	e1.b.Preds[e1.i].i = 0
	b.Likely *= -1
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
