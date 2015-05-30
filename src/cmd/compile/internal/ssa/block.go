// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "fmt"

// Block represents a basic block in the control flow graph of a function.
type Block struct {
	// A unique identifier for the block.  The system will attempt to allocate
	// these IDs densely, but no guarantees.
	ID ID

	// The kind of block this is.
	Kind BlockKind

	// Subsequent blocks, if any.  The number and order depend on the block kind.
	// All successors must be distinct (to make phi values in successors unambiguous).
	Succs []*Block

	// Inverse of successors.
	// The order is significant to Phi nodes in the block.
	Preds []*Block
	// TODO: predecessors is a pain to maintain.  Can we somehow order phi
	// arguments by block id and have this field computed explicitly when needed?

	// A value that determines how the block is exited.  Its value depends on the kind
	// of the block.  For instance, a BlockIf has a boolean control value and BlockExit
	// has a memory control value.
	Control *Value

	// The unordered set of Values that define the operation of this block.
	// The list must include the control value, if any. (TODO: need this last condition?)
	// After the scheduling pass, this list is ordered.
	Values []*Value

	// The containing function
	Func *Func

	// Line number for block's control operation
	Line int32
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
	if b.Control != nil {
		s += fmt.Sprintf(" %s", b.Control)
	}
	if len(b.Succs) > 0 {
		s += " ->"
		for _, c := range b.Succs {
			s += " " + c.String()
		}
	}
	return s
}
