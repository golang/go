// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"math"
)

// A Value represents a value in the SSA representation of the program.
// The ID and Type fields must not be modified. The remainder may be modified
// if they preserve the value of the Value (e.g. changing a (mul 2 x) to an (add x x)).
type Value struct {
	// A unique identifier for the value. For performance we allocate these IDs
	// densely starting at 1.  There is no guarantee that there won't be occasional holes, though.
	ID ID

	// The operation that computes this value. See op.go.
	Op Op

	// The type of this value. Normally this will be a Go type, but there
	// are a few other pseudo-types, see type.go.
	Type Type

	// Auxiliary info for this value. The type of this information depends on the opcode and type.
	// AuxInt is used for integer values, Aux is used for other values.
	AuxInt int64
	Aux    interface{}

	// Arguments of this value
	Args []*Value

	// Containing basic block
	Block *Block

	// Source line number
	Line int32

	// Use count. Each appearance in Value.Args and Block.Control counts once.
	Uses int32

	// Storage for the first three args
	argstorage [3]*Value
}

// Examples:
// Opcode          aux   args
//  OpAdd          nil      2
//  OpConst     string      0    string constant
//  OpConst      int64      0    int64 constant
//  OpAddcq      int64      1    amd64 op: v = arg[0] + constant

// short form print. Just v#.
func (v *Value) String() string {
	if v == nil {
		return "nil" // should never happen, but not panicking helps with debugging
	}
	return fmt.Sprintf("v%d", v.ID)
}

func (v *Value) AuxInt8() int8 {
	if opcodeTable[v.Op].auxType != auxInt8 {
		v.Fatalf("op %s doesn't have an int8 aux field", v.Op)
	}
	return int8(v.AuxInt)
}

func (v *Value) AuxInt16() int16 {
	if opcodeTable[v.Op].auxType != auxInt16 {
		v.Fatalf("op %s doesn't have an int16 aux field", v.Op)
	}
	return int16(v.AuxInt)
}

func (v *Value) AuxInt32() int32 {
	if opcodeTable[v.Op].auxType != auxInt32 {
		v.Fatalf("op %s doesn't have an int32 aux field", v.Op)
	}
	return int32(v.AuxInt)
}

func (v *Value) AuxFloat() float64 {
	if opcodeTable[v.Op].auxType != auxFloat32 && opcodeTable[v.Op].auxType != auxFloat64 {
		v.Fatalf("op %s doesn't have a float aux field", v.Op)
	}
	return math.Float64frombits(uint64(v.AuxInt))
}
func (v *Value) AuxValAndOff() ValAndOff {
	if opcodeTable[v.Op].auxType != auxSymValAndOff {
		v.Fatalf("op %s doesn't have a ValAndOff aux field", v.Op)
	}
	return ValAndOff(v.AuxInt)
}

// long form print.  v# = opcode <type> [aux] args [: reg]
func (v *Value) LongString() string {
	s := fmt.Sprintf("v%d = %s", v.ID, v.Op.String())
	s += " <" + v.Type.String() + ">"
	s += v.auxString()
	for _, a := range v.Args {
		s += fmt.Sprintf(" %v", a)
	}
	r := v.Block.Func.RegAlloc
	if int(v.ID) < len(r) && r[v.ID] != nil {
		s += " : " + r[v.ID].Name()
	}
	return s
}

func (v *Value) auxString() string {
	switch opcodeTable[v.Op].auxType {
	case auxBool:
		if v.AuxInt == 0 {
			return " [false]"
		} else {
			return " [true]"
		}
	case auxInt8:
		return fmt.Sprintf(" [%d]", v.AuxInt8())
	case auxInt16:
		return fmt.Sprintf(" [%d]", v.AuxInt16())
	case auxInt32:
		return fmt.Sprintf(" [%d]", v.AuxInt32())
	case auxInt64, auxInt128:
		return fmt.Sprintf(" [%d]", v.AuxInt)
	case auxFloat32, auxFloat64:
		return fmt.Sprintf(" [%g]", v.AuxFloat())
	case auxString:
		return fmt.Sprintf(" {%q}", v.Aux)
	case auxSym:
		if v.Aux != nil {
			return fmt.Sprintf(" {%s}", v.Aux)
		}
	case auxSymOff, auxSymInt32:
		s := ""
		if v.Aux != nil {
			s = fmt.Sprintf(" {%s}", v.Aux)
		}
		if v.AuxInt != 0 {
			s += fmt.Sprintf(" [%v]", v.AuxInt)
		}
		return s
	case auxSymValAndOff:
		s := ""
		if v.Aux != nil {
			s = fmt.Sprintf(" {%s}", v.Aux)
		}
		return s + fmt.Sprintf(" [%s]", v.AuxValAndOff())
	}
	return ""
}

func (v *Value) AddArg(w *Value) {
	if v.Args == nil {
		v.resetArgs() // use argstorage
	}
	v.Args = append(v.Args, w)
	w.Uses++
}
func (v *Value) AddArgs(a ...*Value) {
	if v.Args == nil {
		v.resetArgs() // use argstorage
	}
	v.Args = append(v.Args, a...)
	for _, x := range a {
		x.Uses++
	}
}
func (v *Value) SetArg(i int, w *Value) {
	v.Args[i].Uses--
	v.Args[i] = w
	w.Uses++
}
func (v *Value) RemoveArg(i int) {
	v.Args[i].Uses--
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
	for _, a := range v.Args {
		a.Uses--
	}
	v.argstorage[0] = nil
	v.argstorage[1] = nil
	v.argstorage[2] = nil
	v.Args = v.argstorage[:0]
}

func (v *Value) reset(op Op) {
	v.Op = op
	v.resetArgs()
	v.AuxInt = 0
	v.Aux = nil
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

// isGenericIntConst returns whether v is a generic integer constant.
func (v *Value) isGenericIntConst() bool {
	return v != nil && (v.Op == OpConst64 || v.Op == OpConst32 || v.Op == OpConst16 || v.Op == OpConst8)
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
