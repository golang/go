// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// A Func represents a Go func declaration (or function literal) and
// its body.  This package compiles each Func independently.
type Func struct {
	Config *Config  // architecture information
	Name   string   // e.g. bytes·Compare
	Type   Type     // type signature of the function.
	Blocks []*Block // unordered set of all basic blocks (note: not indexable by ID)
	Entry  *Block   // the entry basic block
	bid    idAlloc  // block ID allocator
	vid    idAlloc  // value ID allocator

	// when register allocation is done, maps value ids to locations
	RegAlloc []Location
	// when stackalloc is done, the size of the stack frame
	FrameSize int64
}

// NumBlocks returns an integer larger than the id of any Block in the Func.
func (f *Func) NumBlocks() int {
	return f.bid.num()
}

// NumValues returns an integer larger than the id of any Value in the Func.
func (f *Func) NumValues() int {
	return f.vid.num()
}

// NewBlock returns a new block of the given kind and appends it to f.Blocks.
func (f *Func) NewBlock(kind BlockKind) *Block {
	b := &Block{
		ID:   f.bid.get(),
		Kind: kind,
		Func: f,
	}
	f.Blocks = append(f.Blocks, b)
	return b
}

// NewValue returns a new value in the block with no arguments.
func (b *Block) NewValue(line int32, op Op, t Type, aux interface{}) *Value {
	v := &Value{
		ID:    b.Func.vid.get(),
		Op:    op,
		Type:  t,
		Aux:   aux,
		Block: b,
	}
	v.Args = v.argstorage[:0]
	b.Values = append(b.Values, v)
	return v
}

// NewValue1 returns a new value in the block with one argument.
func (b *Block) NewValue1(line int32, op Op, t Type, aux interface{}, arg *Value) *Value {
	v := &Value{
		ID:    b.Func.vid.get(),
		Op:    op,
		Type:  t,
		Aux:   aux,
		Block: b,
	}
	v.Args = v.argstorage[:1]
	v.Args[0] = arg
	b.Values = append(b.Values, v)
	return v
}

// NewValue2 returns a new value in the block with two arguments.
func (b *Block) NewValue2(line int32, op Op, t Type, aux interface{}, arg0, arg1 *Value) *Value {
	v := &Value{
		ID:    b.Func.vid.get(),
		Op:    op,
		Type:  t,
		Aux:   aux,
		Block: b,
	}
	v.Args = v.argstorage[:2]
	v.Args[0] = arg0
	v.Args[1] = arg1
	b.Values = append(b.Values, v)
	return v
}

// NewValue3 returns a new value in the block with three arguments.
func (b *Block) NewValue3(line int32, op Op, t Type, aux interface{}, arg0, arg1, arg2 *Value) *Value {
	v := &Value{
		ID:    b.Func.vid.get(),
		Op:    op,
		Type:  t,
		Aux:   aux,
		Block: b,
	}
	v.Args = []*Value{arg0, arg1, arg2}
	b.Values = append(b.Values, v)
	return v
}

// ConstInt returns an int constant representing its argument.
func (f *Func) ConstInt(line int32, t Type, c int64) *Value {
	// TODO: cache?
	return f.Entry.NewValue(line, OpConst, t, c)
}
