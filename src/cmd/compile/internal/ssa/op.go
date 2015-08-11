// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// An Op encodes the specific operation that a Value performs.
// Opcodes' semantics can be modified by the type and aux fields of the Value.
// For instance, OpAdd can be 32 or 64 bit, signed or unsigned, float or complex, depending on Value.Type.
// Semantics of each op are described in the opcode files in gen/*Ops.go.
// There is one file for generic (architecture-independent) ops and one file
// for each architecture.
type Op int32

type opInfo struct {
	name    string
	asm     int
	reg     regInfo
	generic bool // this is a generic (arch-independent) opcode
}

type inputInfo struct {
	idx  int     // index in Args array
	regs regMask // allowed input registers
}

type regInfo struct {
	inputs   []inputInfo // ordered in register allocation order
	clobbers regMask
	outputs  []regMask // NOTE: values can only have 1 output for now.
}
