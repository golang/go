// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"log"
)

// An Op encodes the specific operation that a Value performs.
// Opcodes' semantics can be modified by the type and aux fields of the Value.
// For instance, OpAdd can be 32 or 64 bit, signed or unsigned, float or complex, depending on Value.Type.
// Semantics of each op are described in the opcode files in gen/*Ops.go.
// There is one file for generic (architecture-independent) ops and one file
// for each architecture.
type Op int32

// GlobalOffset represents a fixed offset within a global variable
type GlobalOffset struct {
	Global interface{} // holds a *gc.Sym
	Offset int64
}

// offset adds x to the location specified by g and returns it.
func (g GlobalOffset) offset(x int64) GlobalOffset {
	y := g.Offset
	z := x + y
	if x^y >= 0 && x^z < 0 {
		log.Panicf("offset overflow %d %d\n", x, y)
	}
	return GlobalOffset{g.Global, z}
}

func (g GlobalOffset) String() string {
	return fmt.Sprintf("%v+%d", g.Global, g.Offset)
}

type opInfo struct {
	name    string
	reg     regInfo
	generic bool // this is a generic (arch-independent) opcode
}

type regInfo struct {
	inputs   []regMask
	clobbers regMask
	outputs  []regMask // NOTE: values can only have 1 output for now.
}
