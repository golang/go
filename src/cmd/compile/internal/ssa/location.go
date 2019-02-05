// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"fmt"
)

// A place that an ssa variable can reside.
type Location interface {
	String() string // name to use in assembly templates: AX, 16(SP), ...
}

// A Register is a machine register, like AX.
// They are numbered densely from 0 (for each architecture).
type Register struct {
	num    int32 // dense numbering
	objNum int16 // register number from cmd/internal/obj/$ARCH
	gcNum  int16 // GC register map number (dense numbering of registers that can contain pointers)
	name   string
}

func (r *Register) String() string {
	return r.name
}

// ObjNum returns the register number from cmd/internal/obj/$ARCH that
// corresponds to this register.
func (r *Register) ObjNum() int16 {
	return r.objNum
}

// GCNum returns the runtime GC register index of r, or -1 if this
// register can't contain pointers.
func (r *Register) GCNum() int16 {
	return r.gcNum
}

// A LocalSlot is a location in the stack frame, which identifies and stores
// part or all of a PPARAM, PPARAMOUT, or PAUTO ONAME node.
// It can represent a whole variable, part of a larger stack slot, or part of a
// variable that has been decomposed into multiple stack slots.
// As an example, a string could have the following configurations:
//
//           stack layout              LocalSlots
//
// Optimizations are disabled. s is on the stack and represented in its entirety.
// [ ------- s string ---- ] { N: s, Type: string, Off: 0 }
//
// s was not decomposed, but the SSA operates on its parts individually, so
// there is a LocalSlot for each of its fields that points into the single stack slot.
// [ ------- s string ---- ] { N: s, Type: *uint8, Off: 0 }, {N: s, Type: int, Off: 8}
//
// s was decomposed. Each of its fields is in its own stack slot and has its own LocalSLot.
// [ ptr *uint8 ] [ len int] { N: ptr, Type: *uint8, Off: 0, SplitOf: parent, SplitOffset: 0},
//                           { N: len, Type: int, Off: 0, SplitOf: parent, SplitOffset: 8}
//                           parent = &{N: s, Type: string}
type LocalSlot struct {
	N    GCNode      // an ONAME *gc.Node representing a stack location.
	Type *types.Type // type of slot
	Off  int64       // offset of slot in N

	SplitOf     *LocalSlot // slot is a decomposition of SplitOf
	SplitOffset int64      // .. at this offset.
}

func (s LocalSlot) String() string {
	if s.Off == 0 {
		return fmt.Sprintf("%v[%v]", s.N, s.Type)
	}
	return fmt.Sprintf("%v+%d[%v]", s.N, s.Off, s.Type)
}

type LocPair [2]Location

func (t LocPair) String() string {
	n0, n1 := "nil", "nil"
	if t[0] != nil {
		n0 = t[0].String()
	}
	if t[1] != nil {
		n1 = t[1].String()
	}
	return fmt.Sprintf("<%s,%s>", n0, n1)
}
