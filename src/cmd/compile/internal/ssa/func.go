// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"math"
	"strings"
)

// A Func represents a Go func declaration (or function literal) and
// its body. This package compiles each Func independently.
type Func struct {
	Config     *Config     // architecture information
	pass       *pass       // current pass information (name, options, etc.)
	Name       string      // e.g. bytes·Compare
	Type       Type        // type signature of the function.
	StaticData interface{} // associated static data, untouched by the ssa package
	Blocks     []*Block    // unordered set of all basic blocks (note: not indexable by ID)
	Entry      *Block      // the entry basic block
	bid        idAlloc     // block ID allocator
	vid        idAlloc     // value ID allocator

	scheduled bool // Values in Blocks are in final order
	NoSplit   bool // true if function is marked as nosplit.  Used by schedule check pass.

	// when register allocation is done, maps value ids to locations
	RegAlloc []Location

	// map from LocalSlot to set of Values that we want to store in that slot.
	NamedValues map[LocalSlot][]*Value
	// Names is a copy of NamedValues.Keys. We keep a separate list
	// of keys to make iteration order deterministic.
	Names []LocalSlot

	freeValues *Value // free Values linked by argstorage[0].  All other fields except ID are 0/nil.
	freeBlocks *Block // free Blocks linked by succstorage[0].b.  All other fields except ID are 0/nil.

	cachedPostorder []*Block   // cached postorder traversal
	cachedIdom      []*Block   // cached immediate dominators
	cachedSdom      SparseTree // cached dominator tree
	cachedLoopnest  *loopnest  // cached loop nest information

	constants map[int64][]*Value // constants cache, keyed by constant value; users must check value's Op and Type
}

// NumBlocks returns an integer larger than the id of any Block in the Func.
func (f *Func) NumBlocks() int {
	return f.bid.num()
}

// NumValues returns an integer larger than the id of any Value in the Func.
func (f *Func) NumValues() int {
	return f.vid.num()
}

// newSparseSet returns a sparse set that can store at least up to n integers.
func (f *Func) newSparseSet(n int) *sparseSet {
	for i, scr := range f.Config.scrSparse {
		if scr != nil && scr.cap() >= n {
			f.Config.scrSparse[i] = nil
			scr.clear()
			return scr
		}
	}
	return newSparseSet(n)
}

// retSparseSet returns a sparse set to the config's cache of sparse sets to be reused by f.newSparseSet.
func (f *Func) retSparseSet(ss *sparseSet) {
	for i, scr := range f.Config.scrSparse {
		if scr == nil {
			f.Config.scrSparse[i] = ss
			return
		}
	}
	f.Config.scrSparse = append(f.Config.scrSparse, ss)
}

// newValue allocates a new Value with the given fields and places it at the end of b.Values.
func (f *Func) newValue(op Op, t Type, b *Block, line int32) *Value {
	var v *Value
	if f.freeValues != nil {
		v = f.freeValues
		f.freeValues = v.argstorage[0]
		v.argstorage[0] = nil
	} else {
		ID := f.vid.get()
		if int(ID) < len(f.Config.values) {
			v = &f.Config.values[ID]
		} else {
			v = &Value{ID: ID}
		}
	}
	v.Op = op
	v.Type = t
	v.Block = b
	v.Line = line
	b.Values = append(b.Values, v)
	return v
}

// logPassStat writes a string key and int value as a warning in a
// tab-separated format easily handled by spreadsheets or awk.
// file names, lines, and function names are included to provide enough (?)
// context to allow item-by-item comparisons across runs.
// For example:
// awk 'BEGIN {FS="\t"} $3~/TIME/{sum+=$4} END{print "t(ns)=",sum}' t.log
func (f *Func) LogStat(key string, args ...interface{}) {
	value := ""
	for _, a := range args {
		value += fmt.Sprintf("\t%v", a)
	}
	n := "missing_pass"
	if f.pass != nil {
		n = strings.Replace(f.pass.name, " ", "_", -1)
	}
	f.Config.Warnl(f.Entry.Line, "\t%s\t%s%s\t%s", n, key, value, f.Name)
}

// freeValue frees a value. It must no longer be referenced.
func (f *Func) freeValue(v *Value) {
	if v.Block == nil {
		f.Fatalf("trying to free an already freed value")
	}
	if v.Uses != 0 {
		f.Fatalf("value %s still has %d uses", v, v.Uses)
	}
	// Clear everything but ID (which we reuse).
	id := v.ID

	// Zero argument values might be cached, so remove them there.
	nArgs := opcodeTable[v.Op].argLen
	if nArgs == 0 {
		vv := f.constants[v.AuxInt]
		for i, cv := range vv {
			if v == cv {
				vv[i] = vv[len(vv)-1]
				f.constants[v.AuxInt] = vv[0 : len(vv)-1]
				break
			}
		}
	}
	*v = Value{}
	v.ID = id
	v.argstorage[0] = f.freeValues
	f.freeValues = v
}

// newBlock allocates a new Block of the given kind and places it at the end of f.Blocks.
func (f *Func) NewBlock(kind BlockKind) *Block {
	var b *Block
	if f.freeBlocks != nil {
		b = f.freeBlocks
		f.freeBlocks = b.succstorage[0].b
		b.succstorage[0].b = nil
	} else {
		ID := f.bid.get()
		if int(ID) < len(f.Config.blocks) {
			b = &f.Config.blocks[ID]
		} else {
			b = &Block{ID: ID}
		}
	}
	b.Kind = kind
	b.Func = f
	b.Preds = b.predstorage[:0]
	b.Succs = b.succstorage[:0]
	b.Values = b.valstorage[:0]
	f.Blocks = append(f.Blocks, b)
	f.invalidateCFG()
	return b
}

func (f *Func) freeBlock(b *Block) {
	if b.Func == nil {
		f.Fatalf("trying to free an already freed block")
	}
	// Clear everything but ID (which we reuse).
	id := b.ID
	*b = Block{}
	b.ID = id
	b.succstorage[0].b = f.freeBlocks
	f.freeBlocks = b
}

// NewValue0 returns a new value in the block with no arguments and zero aux values.
func (b *Block) NewValue0(line int32, op Op, t Type) *Value {
	v := b.Func.newValue(op, t, b, line)
	v.AuxInt = 0
	v.Args = v.argstorage[:0]
	return v
}

// NewValue returns a new value in the block with no arguments and an auxint value.
func (b *Block) NewValue0I(line int32, op Op, t Type, auxint int64) *Value {
	v := b.Func.newValue(op, t, b, line)
	v.AuxInt = auxint
	v.Args = v.argstorage[:0]
	return v
}

// NewValue returns a new value in the block with no arguments and an aux value.
func (b *Block) NewValue0A(line int32, op Op, t Type, aux interface{}) *Value {
	if _, ok := aux.(int64); ok {
		// Disallow int64 aux values. They should be in the auxint field instead.
		// Maybe we want to allow this at some point, but for now we disallow it
		// to prevent errors like using NewValue1A instead of NewValue1I.
		b.Fatalf("aux field has int64 type op=%s type=%s aux=%v", op, t, aux)
	}
	v := b.Func.newValue(op, t, b, line)
	v.AuxInt = 0
	v.Aux = aux
	v.Args = v.argstorage[:0]
	return v
}

// NewValue returns a new value in the block with no arguments and both an auxint and aux values.
func (b *Block) NewValue0IA(line int32, op Op, t Type, auxint int64, aux interface{}) *Value {
	v := b.Func.newValue(op, t, b, line)
	v.AuxInt = auxint
	v.Aux = aux
	v.Args = v.argstorage[:0]
	return v
}

// NewValue1 returns a new value in the block with one argument and zero aux values.
func (b *Block) NewValue1(line int32, op Op, t Type, arg *Value) *Value {
	v := b.Func.newValue(op, t, b, line)
	v.AuxInt = 0
	v.Args = v.argstorage[:1]
	v.argstorage[0] = arg
	arg.Uses++
	return v
}

// NewValue1I returns a new value in the block with one argument and an auxint value.
func (b *Block) NewValue1I(line int32, op Op, t Type, auxint int64, arg *Value) *Value {
	v := b.Func.newValue(op, t, b, line)
	v.AuxInt = auxint
	v.Args = v.argstorage[:1]
	v.argstorage[0] = arg
	arg.Uses++
	return v
}

// NewValue1A returns a new value in the block with one argument and an aux value.
func (b *Block) NewValue1A(line int32, op Op, t Type, aux interface{}, arg *Value) *Value {
	v := b.Func.newValue(op, t, b, line)
	v.AuxInt = 0
	v.Aux = aux
	v.Args = v.argstorage[:1]
	v.argstorage[0] = arg
	arg.Uses++
	return v
}

// NewValue1IA returns a new value in the block with one argument and both an auxint and aux values.
func (b *Block) NewValue1IA(line int32, op Op, t Type, auxint int64, aux interface{}, arg *Value) *Value {
	v := b.Func.newValue(op, t, b, line)
	v.AuxInt = auxint
	v.Aux = aux
	v.Args = v.argstorage[:1]
	v.argstorage[0] = arg
	arg.Uses++
	return v
}

// NewValue2 returns a new value in the block with two arguments and zero aux values.
func (b *Block) NewValue2(line int32, op Op, t Type, arg0, arg1 *Value) *Value {
	v := b.Func.newValue(op, t, b, line)
	v.AuxInt = 0
	v.Args = v.argstorage[:2]
	v.argstorage[0] = arg0
	v.argstorage[1] = arg1
	arg0.Uses++
	arg1.Uses++
	return v
}

// NewValue2I returns a new value in the block with two arguments and an auxint value.
func (b *Block) NewValue2I(line int32, op Op, t Type, auxint int64, arg0, arg1 *Value) *Value {
	v := b.Func.newValue(op, t, b, line)
	v.AuxInt = auxint
	v.Args = v.argstorage[:2]
	v.argstorage[0] = arg0
	v.argstorage[1] = arg1
	arg0.Uses++
	arg1.Uses++
	return v
}

// NewValue3 returns a new value in the block with three arguments and zero aux values.
func (b *Block) NewValue3(line int32, op Op, t Type, arg0, arg1, arg2 *Value) *Value {
	v := b.Func.newValue(op, t, b, line)
	v.AuxInt = 0
	v.Args = v.argstorage[:3]
	v.argstorage[0] = arg0
	v.argstorage[1] = arg1
	v.argstorage[2] = arg2
	arg0.Uses++
	arg1.Uses++
	arg2.Uses++
	return v
}

// NewValue3I returns a new value in the block with three arguments and an auxint value.
func (b *Block) NewValue3I(line int32, op Op, t Type, auxint int64, arg0, arg1, arg2 *Value) *Value {
	v := b.Func.newValue(op, t, b, line)
	v.AuxInt = auxint
	v.Args = v.argstorage[:3]
	v.argstorage[0] = arg0
	v.argstorage[1] = arg1
	v.argstorage[2] = arg2
	arg0.Uses++
	arg1.Uses++
	arg2.Uses++
	return v
}

// NewValue4 returns a new value in the block with four arguments and zero aux values.
func (b *Block) NewValue4(line int32, op Op, t Type, arg0, arg1, arg2, arg3 *Value) *Value {
	v := b.Func.newValue(op, t, b, line)
	v.AuxInt = 0
	v.Args = []*Value{arg0, arg1, arg2, arg3}
	arg0.Uses++
	arg1.Uses++
	arg2.Uses++
	arg3.Uses++
	return v
}

// constVal returns a constant value for c.
func (f *Func) constVal(line int32, op Op, t Type, c int64, setAux bool) *Value {
	if f.constants == nil {
		f.constants = make(map[int64][]*Value)
	}
	vv := f.constants[c]
	for _, v := range vv {
		if v.Op == op && v.Type.Compare(t) == CMPeq {
			if setAux && v.AuxInt != c {
				panic(fmt.Sprintf("cached const %s should have AuxInt of %d", v.LongString(), c))
			}
			return v
		}
	}
	var v *Value
	if setAux {
		v = f.Entry.NewValue0I(line, op, t, c)
	} else {
		v = f.Entry.NewValue0(line, op, t)
	}
	f.constants[c] = append(vv, v)
	return v
}

// These magic auxint values let us easily cache non-numeric constants
// using the same constants map while making collisions unlikely.
// These values are unlikely to occur in regular code and
// are easy to grep for in case of bugs.
const (
	constSliceMagic       = 1122334455
	constInterfaceMagic   = 2233445566
	constNilMagic         = 3344556677
	constEmptyStringMagic = 4455667788
)

// ConstInt returns an int constant representing its argument.
func (f *Func) ConstBool(line int32, t Type, c bool) *Value {
	i := int64(0)
	if c {
		i = 1
	}
	return f.constVal(line, OpConstBool, t, i, true)
}
func (f *Func) ConstInt8(line int32, t Type, c int8) *Value {
	return f.constVal(line, OpConst8, t, int64(c), true)
}
func (f *Func) ConstInt16(line int32, t Type, c int16) *Value {
	return f.constVal(line, OpConst16, t, int64(c), true)
}
func (f *Func) ConstInt32(line int32, t Type, c int32) *Value {
	return f.constVal(line, OpConst32, t, int64(c), true)
}
func (f *Func) ConstInt64(line int32, t Type, c int64) *Value {
	return f.constVal(line, OpConst64, t, c, true)
}
func (f *Func) ConstFloat32(line int32, t Type, c float64) *Value {
	return f.constVal(line, OpConst32F, t, int64(math.Float64bits(float64(float32(c)))), true)
}
func (f *Func) ConstFloat64(line int32, t Type, c float64) *Value {
	return f.constVal(line, OpConst64F, t, int64(math.Float64bits(c)), true)
}

func (f *Func) ConstSlice(line int32, t Type) *Value {
	return f.constVal(line, OpConstSlice, t, constSliceMagic, false)
}
func (f *Func) ConstInterface(line int32, t Type) *Value {
	return f.constVal(line, OpConstInterface, t, constInterfaceMagic, false)
}
func (f *Func) ConstNil(line int32, t Type) *Value {
	return f.constVal(line, OpConstNil, t, constNilMagic, false)
}
func (f *Func) ConstEmptyString(line int32, t Type) *Value {
	v := f.constVal(line, OpConstString, t, constEmptyStringMagic, false)
	v.Aux = ""
	return v
}

func (f *Func) Logf(msg string, args ...interface{})   { f.Config.Logf(msg, args...) }
func (f *Func) Log() bool                              { return f.Config.Log() }
func (f *Func) Fatalf(msg string, args ...interface{}) { f.Config.Fatalf(f.Entry.Line, msg, args...) }

func (f *Func) Free() {
	// Clear cached CFG info.
	f.invalidateCFG()

	// Clear values.
	n := f.vid.num()
	if n > len(f.Config.values) {
		n = len(f.Config.values)
	}
	for i := 1; i < n; i++ {
		f.Config.values[i] = Value{}
		f.Config.values[i].ID = ID(i)
	}

	// Clear blocks.
	n = f.bid.num()
	if n > len(f.Config.blocks) {
		n = len(f.Config.blocks)
	}
	for i := 1; i < n; i++ {
		f.Config.blocks[i] = Block{}
		f.Config.blocks[i].ID = ID(i)
	}

	// Unregister from config.
	if f.Config.curFunc != f {
		f.Fatalf("free of function which isn't the last one allocated")
	}
	f.Config.curFunc = nil
	*f = Func{} // just in case
}

// postorder returns the reachable blocks in f in a postorder traversal.
func (f *Func) postorder() []*Block {
	if f.cachedPostorder == nil {
		f.cachedPostorder = postorder(f)
	}
	return f.cachedPostorder
}

// Idom returns a map from block ID to the immediate dominator of that block.
// f.Entry.ID maps to nil. Unreachable blocks map to nil as well.
func (f *Func) Idom() []*Block {
	if f.cachedIdom == nil {
		f.cachedIdom = dominators(f)
	}
	return f.cachedIdom
}

// sdom returns a sparse tree representing the dominator relationships
// among the blocks of f.
func (f *Func) sdom() SparseTree {
	if f.cachedSdom == nil {
		f.cachedSdom = newSparseTree(f, f.Idom())
	}
	return f.cachedSdom
}

// loopnest returns the loop nest information for f.
func (f *Func) loopnest() *loopnest {
	if f.cachedLoopnest == nil {
		f.cachedLoopnest = loopnestfor(f)
	}
	return f.cachedLoopnest
}

// invalidateCFG tells f that its CFG has changed.
func (f *Func) invalidateCFG() {
	f.cachedPostorder = nil
	f.cachedIdom = nil
	f.cachedSdom = nil
	f.cachedLoopnest = nil
}
