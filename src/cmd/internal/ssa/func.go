// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// A Func represents a Go func declaration (or function literal) and
// its body.  This package compiles each Func independently.
type Func struct {
	Name   string   // e.g. bytes·Compare
	Type   Type     // type signature of the function.
	Blocks []*Block // unordered set of all basic blocks (note: not indexable by ID)
	Entry  *Block   // the entry basic block
	bid    idAlloc  // block ID allocator
	vid    idAlloc  // value ID allocator

	// when register allocation is done, maps value ids to locations
	RegAlloc []Location
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
func (b *Block) NewValue(op Op, t Type, aux interface{}) *Value {
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

// ConstInt returns an int constant representing its argument.
func (f *Func) ConstInt(c int64) *Value {
	// TODO: cache?
	// TODO: different types?
	return f.Entry.NewValue(OpConst, TypeInt64, c)
}
