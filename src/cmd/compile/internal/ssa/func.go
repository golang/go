// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"math"
	"sync"
)

// A Func represents a Go func declaration (or function literal) and
// its body.  This package compiles each Func independently.
type Func struct {
	Config     *Config     // architecture information
	Name       string      // e.g. bytes·Compare
	Type       Type        // type signature of the function.
	StaticData interface{} // associated static data, untouched by the ssa package
	Blocks     []*Block    // unordered set of all basic blocks (note: not indexable by ID)
	Entry      *Block      // the entry basic block
	bid        idAlloc     // block ID allocator
	vid        idAlloc     // value ID allocator

	scheduled bool // Values in Blocks are in final order

	// when register allocation is done, maps value ids to locations
	RegAlloc []Location

	// map from *gc.Node to set of Values that represent that Node.
	// The Node must be an ONAME with PPARAM, PPARAMOUT, or PAUTO class.
	NamedValues map[GCNode][]*Value
	// Names is a copy of NamedValues.Keys.  We keep a separate list
	// of keys to make iteration order deterministic.
	Names []GCNode
}

// NumBlocks returns an integer larger than the id of any Block in the Func.
func (f *Func) NumBlocks() int {
	return f.bid.num()
}

// NumValues returns an integer larger than the id of any Value in the Func.
func (f *Func) NumValues() int {
	return f.vid.num()
}

const (
	blockSize = 100
)

// blockPool provides a contiguous array of Blocks which
// improves the speed of traversing dominator trees.
type blockPool struct {
	blocks []Block
	mu     sync.Mutex
}

func (bp *blockPool) newBlock() *Block {
	bp.mu.Lock()
	defer bp.mu.Unlock()

	if len(bp.blocks) == 0 {
		bp.blocks = make([]Block, blockSize, blockSize)
	}

	res := &bp.blocks[0]
	bp.blocks = bp.blocks[1:]
	return res
}

var bp blockPool

// NewBlock returns a new block of the given kind and appends it to f.Blocks.
func (f *Func) NewBlock(kind BlockKind) *Block {
	b := bp.newBlock()
	b.ID = f.bid.get()
	b.Kind = kind
	b.Func = f
	f.Blocks = append(f.Blocks, b)
	return b
}

// NewValue0 returns a new value in the block with no arguments and zero aux values.
func (b *Block) NewValue0(line int32, op Op, t Type) *Value {
	v := &Value{
		ID:    b.Func.vid.get(),
		Op:    op,
		Type:  t,
		Block: b,
		Line:  line,
	}
	v.Args = v.argstorage[:0]
	b.Values = append(b.Values, v)
	return v
}

// NewValue returns a new value in the block with no arguments and an auxint value.
func (b *Block) NewValue0I(line int32, op Op, t Type, auxint int64) *Value {
	v := &Value{
		ID:     b.Func.vid.get(),
		Op:     op,
		Type:   t,
		AuxInt: auxint,
		Block:  b,
		Line:   line,
	}
	v.Args = v.argstorage[:0]
	b.Values = append(b.Values, v)
	return v
}

// NewValue returns a new value in the block with no arguments and an aux value.
func (b *Block) NewValue0A(line int32, op Op, t Type, aux interface{}) *Value {
	if _, ok := aux.(int64); ok {
		// Disallow int64 aux values.  They should be in the auxint field instead.
		// Maybe we want to allow this at some point, but for now we disallow it
		// to prevent errors like using NewValue1A instead of NewValue1I.
		b.Fatalf("aux field has int64 type op=%s type=%s aux=%v", op, t, aux)
	}
	v := &Value{
		ID:    b.Func.vid.get(),
		Op:    op,
		Type:  t,
		Aux:   aux,
		Block: b,
		Line:  line,
	}
	v.Args = v.argstorage[:0]
	b.Values = append(b.Values, v)
	return v
}

// NewValue returns a new value in the block with no arguments and both an auxint and aux values.
func (b *Block) NewValue0IA(line int32, op Op, t Type, auxint int64, aux interface{}) *Value {
	v := &Value{
		ID:     b.Func.vid.get(),
		Op:     op,
		Type:   t,
		AuxInt: auxint,
		Aux:    aux,
		Block:  b,
		Line:   line,
	}
	v.Args = v.argstorage[:0]
	b.Values = append(b.Values, v)
	return v
}

// NewValue1 returns a new value in the block with one argument and zero aux values.
func (b *Block) NewValue1(line int32, op Op, t Type, arg *Value) *Value {
	v := &Value{
		ID:    b.Func.vid.get(),
		Op:    op,
		Type:  t,
		Block: b,
		Line:  line,
	}
	v.Args = v.argstorage[:1]
	v.Args[0] = arg
	b.Values = append(b.Values, v)
	return v
}

// NewValue1I returns a new value in the block with one argument and an auxint value.
func (b *Block) NewValue1I(line int32, op Op, t Type, auxint int64, arg *Value) *Value {
	v := &Value{
		ID:     b.Func.vid.get(),
		Op:     op,
		Type:   t,
		AuxInt: auxint,
		Block:  b,
		Line:   line,
	}
	v.Args = v.argstorage[:1]
	v.Args[0] = arg
	b.Values = append(b.Values, v)
	return v
}

// NewValue1A returns a new value in the block with one argument and an aux value.
func (b *Block) NewValue1A(line int32, op Op, t Type, aux interface{}, arg *Value) *Value {
	v := &Value{
		ID:    b.Func.vid.get(),
		Op:    op,
		Type:  t,
		Aux:   aux,
		Block: b,
		Line:  line,
	}
	v.Args = v.argstorage[:1]
	v.Args[0] = arg
	b.Values = append(b.Values, v)
	return v
}

// NewValue1IA returns a new value in the block with one argument and both an auxint and aux values.
func (b *Block) NewValue1IA(line int32, op Op, t Type, auxint int64, aux interface{}, arg *Value) *Value {
	v := &Value{
		ID:     b.Func.vid.get(),
		Op:     op,
		Type:   t,
		AuxInt: auxint,
		Aux:    aux,
		Block:  b,
		Line:   line,
	}
	v.Args = v.argstorage[:1]
	v.Args[0] = arg
	b.Values = append(b.Values, v)
	return v
}

// NewValue2 returns a new value in the block with two arguments and zero aux values.
func (b *Block) NewValue2(line int32, op Op, t Type, arg0, arg1 *Value) *Value {
	v := &Value{
		ID:    b.Func.vid.get(),
		Op:    op,
		Type:  t,
		Block: b,
		Line:  line,
	}
	v.Args = v.argstorage[:2]
	v.Args[0] = arg0
	v.Args[1] = arg1
	b.Values = append(b.Values, v)
	return v
}

// NewValue2I returns a new value in the block with two arguments and an auxint value.
func (b *Block) NewValue2I(line int32, op Op, t Type, aux int64, arg0, arg1 *Value) *Value {
	v := &Value{
		ID:     b.Func.vid.get(),
		Op:     op,
		Type:   t,
		AuxInt: aux,
		Block:  b,
		Line:   line,
	}
	v.Args = v.argstorage[:2]
	v.Args[0] = arg0
	v.Args[1] = arg1
	b.Values = append(b.Values, v)
	return v
}

// NewValue3 returns a new value in the block with three arguments and zero aux values.
func (b *Block) NewValue3(line int32, op Op, t Type, arg0, arg1, arg2 *Value) *Value {
	v := &Value{
		ID:    b.Func.vid.get(),
		Op:    op,
		Type:  t,
		Block: b,
		Line:  line,
	}
	v.Args = []*Value{arg0, arg1, arg2}
	b.Values = append(b.Values, v)
	return v
}

// NewValue3I returns a new value in the block with three arguments and an auxint value.
func (b *Block) NewValue3I(line int32, op Op, t Type, aux int64, arg0, arg1, arg2 *Value) *Value {
	v := &Value{
		ID:     b.Func.vid.get(),
		Op:     op,
		Type:   t,
		AuxInt: aux,
		Block:  b,
		Line:   line,
	}
	v.Args = []*Value{arg0, arg1, arg2}
	b.Values = append(b.Values, v)
	return v
}

// ConstInt returns an int constant representing its argument.
func (f *Func) ConstBool(line int32, t Type, c bool) *Value {
	// TODO: cache?
	i := int64(0)
	if c {
		i = 1
	}
	return f.Entry.NewValue0I(line, OpConstBool, t, i)
}
func (f *Func) ConstInt8(line int32, t Type, c int8) *Value {
	// TODO: cache?
	return f.Entry.NewValue0I(line, OpConst8, t, int64(c))
}
func (f *Func) ConstInt16(line int32, t Type, c int16) *Value {
	// TODO: cache?
	return f.Entry.NewValue0I(line, OpConst16, t, int64(c))
}
func (f *Func) ConstInt32(line int32, t Type, c int32) *Value {
	// TODO: cache?
	return f.Entry.NewValue0I(line, OpConst32, t, int64(c))
}
func (f *Func) ConstInt64(line int32, t Type, c int64) *Value {
	// TODO: cache?
	return f.Entry.NewValue0I(line, OpConst64, t, c)
}
func (f *Func) ConstIntPtr(line int32, t Type, c int64) *Value {
	// TODO: cache?
	return f.Entry.NewValue0I(line, OpConstPtr, t, c)
}
func (f *Func) ConstFloat32(line int32, t Type, c float64) *Value {
	// TODO: cache?
	return f.Entry.NewValue0I(line, OpConst32F, t, int64(math.Float64bits(c)))
}
func (f *Func) ConstFloat64(line int32, t Type, c float64) *Value {
	// TODO: cache?
	return f.Entry.NewValue0I(line, OpConst64F, t, int64(math.Float64bits(c)))
}

func (f *Func) Logf(msg string, args ...interface{})           { f.Config.Logf(msg, args...) }
func (f *Func) Fatalf(msg string, args ...interface{})         { f.Config.Fatalf(msg, args...) }
func (f *Func) Unimplementedf(msg string, args ...interface{}) { f.Config.Unimplementedf(msg, args...) }
