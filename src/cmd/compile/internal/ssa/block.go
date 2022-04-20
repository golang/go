// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/internal/src"
	"fmt"
)

// Block represents a basic block in the control flow graph of a function.
type Block struct {
	// A unique identifier for the block. The system will attempt to allocate
	// these IDs densely, but no guarantees.
	ID ID

	// Source position for block's control operation
	Pos src.XPos

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

	// A list of values that determine how the block is exited. The number
	// and type of control values depends on the Kind of the block. For
	// instance, a BlockIf has a single boolean control value and BlockExit
	// has a single memory control value.
	//
	// The ControlValues() method may be used to get a slice with the non-nil
	// control values that can be ranged over.
	//
	// Controls[1] must be nil if Controls[0] is nil.
	Controls [2]*Value

	// Auxiliary info for the block. Its value depends on the Kind.
	Aux    Aux
	AuxInt int64

	// The unordered set of Values that define the operation of this block.
	// After the scheduling pass, this list is ordered.
	Values []*Value

	// The containing function
	Func *Func

	// Storage for Succs, Preds and Values.
	succstorage [2]Edge
	predstorage [4]Edge
	valstorage  [9]*Value
}

// Edge represents a CFG edge.
// Example edges for b branching to either c or d.
// (c and d have other predecessors.)
//
//	b.Succs = [{c,3}, {d,1}]
//	c.Preds = [?, ?, ?, {b,0}]
//	d.Preds = [?, {b,1}, ?]
//
// These indexes allow us to edit the CFG in constant time.
// In addition, it informs phi ops in degenerate cases like:
//
//	b:
//	   if k then c else c
//	c:
//	   v = Phi(x, y)
//
// Then the indexes tell you whether x is chosen from
// the if or else branch from b.
//
//	b.Succs = [{c,0},{c,1}]
//	c.Preds = [{b,0},{b,1}]
//
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
func (e Edge) Index() int {
	return e.i
}
func (e Edge) String() string {
	return fmt.Sprintf("{%v,%d}", e.b, e.i)
}

// BlockKind is the kind of SSA block.
//
//	  kind          controls        successors
//	------------------------------------------
//	  Exit      [return mem]                []
//	 Plain                []            [next]
//	    If   [boolean Value]      [then, else]
//	 Defer             [mem]  [nopanic, panic]  (control opcode should be OpStaticCall to runtime.deferproc)
type BlockKind int8

// short form print
func (b *Block) String() string {
	return fmt.Sprintf("b%d", b.ID)
}

// long form print
func (b *Block) LongString() string {
	s := b.Kind.String()
	if b.Aux != nil {
		s += fmt.Sprintf(" {%s}", b.Aux)
	}
	if t := b.AuxIntString(); t != "" {
		s += fmt.Sprintf(" [%s]", t)
	}
	for _, c := range b.ControlValues() {
		s += fmt.Sprintf(" %s", c)
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

// NumControls returns the number of non-nil control values the
// block has.
func (b *Block) NumControls() int {
	if b.Controls[0] == nil {
		return 0
	}
	if b.Controls[1] == nil {
		return 1
	}
	return 2
}

// ControlValues returns a slice containing the non-nil control
// values of the block. The index of each control value will be
// the same as it is in the Controls property and can be used
// in ReplaceControl calls.
func (b *Block) ControlValues() []*Value {
	if b.Controls[0] == nil {
		return b.Controls[:0]
	}
	if b.Controls[1] == nil {
		return b.Controls[:1]
	}
	return b.Controls[:2]
}

// SetControl removes all existing control values and then adds
// the control value provided. The number of control values after
// a call to SetControl will always be 1.
func (b *Block) SetControl(v *Value) {
	b.ResetControls()
	b.Controls[0] = v
	v.Uses++
}

// ResetControls sets the number of controls for the block to 0.
func (b *Block) ResetControls() {
	if b.Controls[0] != nil {
		b.Controls[0].Uses--
	}
	if b.Controls[1] != nil {
		b.Controls[1].Uses--
	}
	b.Controls = [2]*Value{} // reset both controls to nil
}

// AddControl appends a control value to the existing list of control values.
func (b *Block) AddControl(v *Value) {
	i := b.NumControls()
	b.Controls[i] = v // panics if array is full
	v.Uses++
}

// ReplaceControl exchanges the existing control value at the index provided
// for the new value. The index must refer to a valid control value.
func (b *Block) ReplaceControl(i int, v *Value) {
	b.Controls[i].Uses--
	b.Controls[i] = v
	v.Uses++
}

// CopyControls replaces the controls for this block with those from the
// provided block. The provided block is not modified.
func (b *Block) CopyControls(from *Block) {
	if b == from {
		return
	}
	b.ResetControls()
	for _, c := range from.ControlValues() {
		b.AddControl(c)
	}
}

// Reset sets the block to the provided kind and clears all the blocks control
// and auxiliary values. Other properties of the block, such as its successors,
// predecessors and values are left unmodified.
func (b *Block) Reset(kind BlockKind) {
	b.Kind = kind
	b.ResetControls()
	b.Aux = nil
	b.AuxInt = 0
}

// resetWithControl resets b and adds control v.
// It is equivalent to b.Reset(kind); b.AddControl(v),
// except that it is one call instead of two and avoids a bounds check.
// It is intended for use by rewrite rules, where this matters.
func (b *Block) resetWithControl(kind BlockKind, v *Value) {
	b.Kind = kind
	b.ResetControls()
	b.Aux = nil
	b.AuxInt = 0
	b.Controls[0] = v
	v.Uses++
}

// resetWithControl2 resets b and adds controls v and w.
// It is equivalent to b.Reset(kind); b.AddControl(v); b.AddControl(w),
// except that it is one call instead of three and avoids two bounds checks.
// It is intended for use by rewrite rules, where this matters.
func (b *Block) resetWithControl2(kind BlockKind, v, w *Value) {
	b.Kind = kind
	b.ResetControls()
	b.Aux = nil
	b.AuxInt = 0
	b.Controls[0] = v
	b.Controls[1] = w
	v.Uses++
	w.Uses++
}

// truncateValues truncates b.Values at the ith element, zeroing subsequent elements.
// The values in b.Values after i must already have had their args reset,
// to maintain correct value uses counts.
func (b *Block) truncateValues(i int) {
	tail := b.Values[i:]
	for j := range tail {
		tail[j] = nil
	}
	b.Values = b.Values[:i]
}

// AddEdgeTo adds an edge from block b to block c. Used during building of the
// SSA graph; do not use on an already-completed SSA graph.
func (b *Block) AddEdgeTo(c *Block) {
	i := len(b.Succs)
	j := len(c.Preds)
	b.Succs = append(b.Succs, Edge{c, j})
	c.Preds = append(c.Preds, Edge{b, i})
	b.Func.invalidateCFG()
}

// removePred removes the ith input edge from b.
// It is the responsibility of the caller to remove
// the corresponding successor edge, and adjust any
// phi values by calling b.removePhiArg(v, i).
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
	b.Func.invalidateCFG()
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
	b.Func.invalidateCFG()
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

// removePhiArg removes the ith arg from phi.
// It must be called after calling b.removePred(i) to
// adjust the corresponding phi value of the block:
//
// b.removePred(i)
// for _, v := range b.Values {
//
//	if v.Op != OpPhi {
//	    continue
//	}
//	b.removeArg(v, i)
//
// }
func (b *Block) removePhiArg(phi *Value, i int) {
	n := len(b.Preds)
	if numPhiArgs := len(phi.Args); numPhiArgs-1 != n {
		b.Fatalf("inconsistent state, num predecessors: %d, num phi args: %d", n, numPhiArgs)
	}
	phi.Args[i].Uses--
	phi.Args[i] = phi.Args[n]
	phi.Args[n] = nil
	phi.Args = phi.Args[:n]
}

// LackingPos indicates whether b is a block whose position should be inherited
// from its successors.  This is true if all the values within it have unreliable positions
// and if it is "plain", meaning that there is no control flow that is also very likely
// to correspond to a well-understood source position.
func (b *Block) LackingPos() bool {
	// Non-plain predecessors are If or Defer, which both (1) have two successors,
	// which might have different line numbers and (2) correspond to statements
	// in the source code that have positions, so this case ought not occur anyway.
	if b.Kind != BlockPlain {
		return false
	}
	if b.Pos != src.NoXPos {
		return false
	}
	for _, v := range b.Values {
		if v.LackingPos() {
			continue
		}
		return false
	}
	return true
}

func (b *Block) AuxIntString() string {
	switch b.Kind.AuxIntType() {
	case "int8":
		return fmt.Sprintf("%v", int8(b.AuxInt))
	case "uint8":
		return fmt.Sprintf("%v", uint8(b.AuxInt))
	default: // type specified but not implemented - print as int64
		return fmt.Sprintf("%v", b.AuxInt)
	case "": // no aux int type
		return ""
	}
}

// likelyBranch reports whether block b is the likely branch of all of its predecessors.
func (b *Block) likelyBranch() bool {
	if len(b.Preds) == 0 {
		return false
	}
	for _, e := range b.Preds {
		p := e.b
		if len(p.Succs) == 1 || len(p.Succs) == 2 && (p.Likely == BranchLikely && p.Succs[0].b == b ||
			p.Likely == BranchUnlikely && p.Succs[1].b == b) {
			continue
		}
		return false
	}
	return true
}

func (b *Block) Logf(msg string, args ...interface{})   { b.Func.Logf(msg, args...) }
func (b *Block) Log() bool                              { return b.Func.Log() }
func (b *Block) Fatalf(msg string, args ...interface{}) { b.Func.Fatalf(msg, args...) }

type BranchPrediction int8

const (
	BranchUnlikely = BranchPrediction(-1)
	BranchUnknown  = BranchPrediction(0)
	BranchLikely   = BranchPrediction(+1)
)
