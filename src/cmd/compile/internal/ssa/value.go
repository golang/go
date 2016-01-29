// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"math"
)

// A Value represents a value in the SSA representation of the program.
// The ID and Type fields must not be modified.  The remainder may be modified
// if they preserve the value of the Value (e.g. changing a (mul 2 x) to an (add x x)).
type Value struct {
	// A unique identifier for the value.  For performance we allocate these IDs
	// densely starting at 1.  There is no guarantee that there won't be occasional holes, though.
	ID ID

	// The operation that computes this value.  See op.go.
	Op Op

	// The type of this value.  Normally this will be a Go type, but there
	// are a few other pseudo-types, see type.go.
	Type Type

	// Auxiliary info for this value.  The type of this information depends on the opcode and type.
	// AuxInt is used for integer values, Aux is used for other values.
	AuxInt int64
	Aux    interface{}

	// Arguments of this value
	Args []*Value

	// Containing basic block
	Block *Block

	// Source line number
	Line int32

	// Storage for the first two args
	argstorage [2]*Value
}

// Examples:
// Opcode          aux   args
//  OpAdd          nil      2
//  OpConst     string      0    string constant
//  OpConst      int64      0    int64 constant
//  OpAddcq      int64      1    amd64 op: v = arg[0] + constant

// short form print.  Just v#.
func (v *Value) String() string {
	if v == nil {
		return "nil" // should never happen, but not panicking helps with debugging
	}
	return fmt.Sprintf("v%d", v.ID)
}

// long form print.  v# = opcode <type> [aux] args [: reg]
func (v *Value) LongString() string {
	s := fmt.Sprintf("v%d = %s", v.ID, v.Op.String())
	s += " <" + v.Type.String() + ">"
	// TODO: use some operator property flags to decide
	// what is encoded in the AuxInt field.
	switch v.Op {
	case OpConst32F, OpConst64F:
		s += fmt.Sprintf(" [%g]", math.Float64frombits(uint64(v.AuxInt)))
	case OpConstBool:
		if v.AuxInt == 0 {
			s += " [false]"
		} else {
			s += " [true]"
		}
	case OpAMD64MOVBstoreconst, OpAMD64MOVWstoreconst, OpAMD64MOVLstoreconst, OpAMD64MOVQstoreconst:
		s += fmt.Sprintf(" [%s]", ValAndOff(v.AuxInt))
	default:
		if v.AuxInt != 0 {
			s += fmt.Sprintf(" [%d]", v.AuxInt)
		}
	}
	if v.Aux != nil {
		if _, ok := v.Aux.(string); ok {
			s += fmt.Sprintf(" {%q}", v.Aux)
		} else {
			s += fmt.Sprintf(" {%v}", v.Aux)
		}
	}
	for _, a := range v.Args {
		s += fmt.Sprintf(" %v", a)
	}
	r := v.Block.Func.RegAlloc
	if int(v.ID) < len(r) && r[v.ID] != nil {
		s += " : " + r[v.ID].Name()
	}
	return s
}

func (v *Value) AddArg(w *Value) {
	if v.Args == nil {
		v.resetArgs() // use argstorage
	}
	v.Args = append(v.Args, w)
}
func (v *Value) AddArgs(a ...*Value) {
	if v.Args == nil {
		v.resetArgs() // use argstorage
	}
	v.Args = append(v.Args, a...)
}
func (v *Value) SetArg(i int, w *Value) {
	v.Args[i] = w
}
func (v *Value) RemoveArg(i int) {
	copy(v.Args[i:], v.Args[i+1:])
	v.Args[len(v.Args)-1] = nil // aid GC
	v.Args = v.Args[:len(v.Args)-1]
}
func (v *Value) SetArgs1(a *Value) {
	v.resetArgs()
	v.AddArg(a)
}
func (v *Value) SetArgs2(a *Value, b *Value) {
	v.resetArgs()
	v.AddArg(a)
	v.AddArg(b)
}

func (v *Value) resetArgs() {
	v.argstorage[0] = nil
	v.argstorage[1] = nil
	v.Args = v.argstorage[:0]
}

// copyInto makes a new value identical to v and adds it to the end of b.
func (v *Value) copyInto(b *Block) *Value {
	c := b.NewValue0(v.Line, v.Op, v.Type)
	c.Aux = v.Aux
	c.AuxInt = v.AuxInt
	c.AddArgs(v.Args...)
	for _, a := range v.Args {
		if a.Type.IsMemory() {
			v.Fatalf("can't move a value with a memory arg %s", v.LongString())
		}
	}
	return c
}

func (v *Value) Logf(msg string, args ...interface{}) { v.Block.Logf(msg, args...) }
func (v *Value) Log() bool                            { return v.Block.Log() }
func (v *Value) Fatalf(msg string, args ...interface{}) {
	v.Block.Func.Config.Fatalf(v.Line, msg, args...)
}
func (v *Value) Unimplementedf(msg string, args ...interface{}) {
	v.Block.Func.Config.Unimplementedf(v.Line, msg, args...)
}

// ExternSymbol is an aux value that encodes a variable's
// constant offset from the static base pointer.
type ExternSymbol struct {
	Typ Type         // Go type
	Sym fmt.Stringer // A *gc.Sym referring to a global variable
	// Note: the offset for an external symbol is not
	// calculated until link time.
}

// ArgSymbol is an aux value that encodes an argument or result
// variable's constant offset from FP (FP = SP + framesize).
type ArgSymbol struct {
	Typ  Type   // Go type
	Node GCNode // A *gc.Node referring to the argument/result variable.
}

// AutoSymbol is an aux value that encodes a local variable's
// constant offset from SP.
type AutoSymbol struct {
	Typ  Type   // Go type
	Node GCNode // A *gc.Node referring to a local (auto) variable.
}

func (s *ExternSymbol) String() string {
	return s.Sym.String()
}

func (s *ArgSymbol) String() string {
	return s.Node.String()
}

func (s *AutoSymbol) String() string {
	return s.Node.String()
}
