// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/abi"
	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"crypto/sha1"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

type writeSyncer interface {
	io.Writer
	Sync() error
}

// A Func represents a Go func declaration (or function literal) and its body.
// This package compiles each Func independently.
// Funcs are single-use; a new Func must be created for every compiled function.
type Func struct {
	Config *Config     // architecture information
	Cache  *Cache      // re-usable cache
	fe     Frontend    // frontend state associated with this Func, callbacks into compiler frontend
	pass   *pass       // current pass information (name, options, etc.)
	Name   string      // e.g. NewFunc or (*Func).NumBlocks (no package prefix)
	Type   *types.Type // type signature of the function.
	Blocks []*Block    // unordered set of all basic blocks (note: not indexable by ID)
	Entry  *Block      // the entry basic block

	bid idAlloc // block ID allocator
	vid idAlloc // value ID allocator

	// Given an environment variable used for debug hash match,
	// what file (if any) receives the yes/no logging?
	logfiles       map[string]writeSyncer
	HTMLWriter     *HTMLWriter    // html writer, for debugging
	DebugTest      bool           // default true unless $GOSSAHASH != ""; as a debugging aid, make new code conditional on this and use GOSSAHASH to binary search for failing cases
	PrintOrHtmlSSA bool           // true if GOSSAFUNC matches, true even if fe.Log() (spew phase results to stdout) is false.  There's an odd dependence on this in debug.go for method logf.
	ruleMatches    map[string]int // number of times countRule was called during compilation for any given string
	ABI0           *abi.ABIConfig // A copy, for no-sync access
	ABI1           *abi.ABIConfig // A copy, for no-sync access
	ABISelf        *abi.ABIConfig // ABI for function being compiled
	ABIDefault     *abi.ABIConfig // ABI for rtcall and other no-parsed-signature/pragma functions.

	scheduled   bool  // Values in Blocks are in final order
	laidout     bool  // Blocks are ordered
	NoSplit     bool  // true if function is marked as nosplit.  Used by schedule check pass.
	dumpFileSeq uint8 // the sequence numbers of dump file. (%s_%02d__%s.dump", funcname, dumpFileSeq, phaseName)

	// when register allocation is done, maps value ids to locations
	RegAlloc []Location

	// map from LocalSlot to set of Values that we want to store in that slot.
	NamedValues map[LocalSlot][]*Value
	// Names is a copy of NamedValues.Keys. We keep a separate list
	// of keys to make iteration order deterministic.
	Names []*LocalSlot
	// Canonicalize root/top-level local slots, and canonicalize their pieces.
	// Because LocalSlot pieces refer to their parents with a pointer, this ensures that equivalent slots really are equal.
	CanonicalLocalSlots  map[LocalSlot]*LocalSlot
	CanonicalLocalSplits map[LocalSlotSplitKey]*LocalSlot

	// RegArgs is a slice of register-memory pairs that must be spilled and unspilled in the uncommon path of function entry.
	RegArgs []Spill
	// AuxCall describing parameters and results for this function.
	OwnAux *AuxCall

	// WBLoads is a list of Blocks that branch on the write
	// barrier flag. Safe-points are disabled from the OpLoad that
	// reads the write-barrier flag until the control flow rejoins
	// below the two successors of this block.
	WBLoads []*Block

	freeValues *Value // free Values linked by argstorage[0].  All other fields except ID are 0/nil.
	freeBlocks *Block // free Blocks linked by succstorage[0].b.  All other fields except ID are 0/nil.

	cachedPostorder  []*Block   // cached postorder traversal
	cachedIdom       []*Block   // cached immediate dominators
	cachedSdom       SparseTree // cached dominator tree
	cachedLoopnest   *loopnest  // cached loop nest information
	cachedLineStarts *xposmap   // cached map/set of xpos to integers

	auxmap    auxmap             // map from aux values to opaque ids used by CSE
	constants map[int64][]*Value // constants cache, keyed by constant value; users must check value's Op and Type
}

type LocalSlotSplitKey struct {
	parent *LocalSlot
	Off    int64       // offset of slot in N
	Type   *types.Type // type of slot
}

// NewFunc returns a new, empty function object.
// Caller must set f.Config and f.Cache before using f.
func NewFunc(fe Frontend) *Func {
	return &Func{fe: fe, NamedValues: make(map[LocalSlot][]*Value), CanonicalLocalSlots: make(map[LocalSlot]*LocalSlot), CanonicalLocalSplits: make(map[LocalSlotSplitKey]*LocalSlot)}
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
	for i, scr := range f.Cache.scrSparseSet {
		if scr != nil && scr.cap() >= n {
			f.Cache.scrSparseSet[i] = nil
			scr.clear()
			return scr
		}
	}
	return newSparseSet(n)
}

// retSparseSet returns a sparse set to the config's cache of sparse
// sets to be reused by f.newSparseSet.
func (f *Func) retSparseSet(ss *sparseSet) {
	for i, scr := range f.Cache.scrSparseSet {
		if scr == nil {
			f.Cache.scrSparseSet[i] = ss
			return
		}
	}
	f.Cache.scrSparseSet = append(f.Cache.scrSparseSet, ss)
}

// newSparseMap returns a sparse map that can store at least up to n integers.
func (f *Func) newSparseMap(n int) *sparseMap {
	for i, scr := range f.Cache.scrSparseMap {
		if scr != nil && scr.cap() >= n {
			f.Cache.scrSparseMap[i] = nil
			scr.clear()
			return scr
		}
	}
	return newSparseMap(n)
}

// retSparseMap returns a sparse map to the config's cache of sparse
// sets to be reused by f.newSparseMap.
func (f *Func) retSparseMap(ss *sparseMap) {
	for i, scr := range f.Cache.scrSparseMap {
		if scr == nil {
			f.Cache.scrSparseMap[i] = ss
			return
		}
	}
	f.Cache.scrSparseMap = append(f.Cache.scrSparseMap, ss)
}

// newPoset returns a new poset from the internal cache
func (f *Func) newPoset() *poset {
	if len(f.Cache.scrPoset) > 0 {
		po := f.Cache.scrPoset[len(f.Cache.scrPoset)-1]
		f.Cache.scrPoset = f.Cache.scrPoset[:len(f.Cache.scrPoset)-1]
		return po
	}
	return newPoset()
}

// retPoset returns a poset to the internal cache
func (f *Func) retPoset(po *poset) {
	f.Cache.scrPoset = append(f.Cache.scrPoset, po)
}

// newDeadcodeLive returns a slice for the
// deadcode pass to use to indicate which values are live.
func (f *Func) newDeadcodeLive() []bool {
	r := f.Cache.deadcode.live
	f.Cache.deadcode.live = nil
	return r
}

// retDeadcodeLive returns a deadcode live value slice for re-use.
func (f *Func) retDeadcodeLive(live []bool) {
	f.Cache.deadcode.live = live
}

// newDeadcodeLiveOrderStmts returns a slice for the
// deadcode pass to use to indicate which values
// need special treatment for statement boundaries.
func (f *Func) newDeadcodeLiveOrderStmts() []*Value {
	r := f.Cache.deadcode.liveOrderStmts
	f.Cache.deadcode.liveOrderStmts = nil
	return r
}

// retDeadcodeLiveOrderStmts returns a deadcode liveOrderStmts slice for re-use.
func (f *Func) retDeadcodeLiveOrderStmts(liveOrderStmts []*Value) {
	f.Cache.deadcode.liveOrderStmts = liveOrderStmts
}

func (f *Func) localSlotAddr(slot LocalSlot) *LocalSlot {
	a, ok := f.CanonicalLocalSlots[slot]
	if !ok {
		a = new(LocalSlot)
		*a = slot // don't escape slot
		f.CanonicalLocalSlots[slot] = a
	}
	return a
}

func (f *Func) SplitString(name *LocalSlot) (*LocalSlot, *LocalSlot) {
	ptrType := types.NewPtr(types.Types[types.TUINT8])
	lenType := types.Types[types.TINT]
	// Split this string up into two separate variables.
	p := f.SplitSlot(name, ".ptr", 0, ptrType)
	l := f.SplitSlot(name, ".len", ptrType.Size(), lenType)
	return p, l
}

func (f *Func) SplitInterface(name *LocalSlot) (*LocalSlot, *LocalSlot) {
	n := name.N
	u := types.Types[types.TUINTPTR]
	t := types.NewPtr(types.Types[types.TUINT8])
	// Split this interface up into two separate variables.
	sfx := ".itab"
	if n.Type().IsEmptyInterface() {
		sfx = ".type"
	}
	c := f.SplitSlot(name, sfx, 0, u) // see comment in typebits.Set
	d := f.SplitSlot(name, ".data", u.Size(), t)
	return c, d
}

func (f *Func) SplitSlice(name *LocalSlot) (*LocalSlot, *LocalSlot, *LocalSlot) {
	ptrType := types.NewPtr(name.Type.Elem())
	lenType := types.Types[types.TINT]
	p := f.SplitSlot(name, ".ptr", 0, ptrType)
	l := f.SplitSlot(name, ".len", ptrType.Size(), lenType)
	c := f.SplitSlot(name, ".cap", ptrType.Size()+lenType.Size(), lenType)
	return p, l, c
}

func (f *Func) SplitComplex(name *LocalSlot) (*LocalSlot, *LocalSlot) {
	s := name.Type.Size() / 2
	var t *types.Type
	if s == 8 {
		t = types.Types[types.TFLOAT64]
	} else {
		t = types.Types[types.TFLOAT32]
	}
	r := f.SplitSlot(name, ".real", 0, t)
	i := f.SplitSlot(name, ".imag", t.Size(), t)
	return r, i
}

func (f *Func) SplitInt64(name *LocalSlot) (*LocalSlot, *LocalSlot) {
	var t *types.Type
	if name.Type.IsSigned() {
		t = types.Types[types.TINT32]
	} else {
		t = types.Types[types.TUINT32]
	}
	if f.Config.BigEndian {
		return f.SplitSlot(name, ".hi", 0, t), f.SplitSlot(name, ".lo", t.Size(), types.Types[types.TUINT32])
	}
	return f.SplitSlot(name, ".hi", t.Size(), t), f.SplitSlot(name, ".lo", 0, types.Types[types.TUINT32])
}

func (f *Func) SplitStruct(name *LocalSlot, i int) *LocalSlot {
	st := name.Type
	return f.SplitSlot(name, st.FieldName(i), st.FieldOff(i), st.FieldType(i))
}
func (f *Func) SplitArray(name *LocalSlot) *LocalSlot {
	n := name.N
	at := name.Type
	if at.NumElem() != 1 {
		base.FatalfAt(n.Pos(), "bad array size")
	}
	et := at.Elem()
	return f.SplitSlot(name, "[0]", 0, et)
}

func (f *Func) SplitSlot(name *LocalSlot, sfx string, offset int64, t *types.Type) *LocalSlot {
	lssk := LocalSlotSplitKey{name, offset, t}
	if als, ok := f.CanonicalLocalSplits[lssk]; ok {
		return als
	}
	// Note: the _ field may appear several times.  But
	// have no fear, identically-named but distinct Autos are
	// ok, albeit maybe confusing for a debugger.
	ls := f.fe.SplitSlot(name, sfx, offset, t)
	f.CanonicalLocalSplits[lssk] = &ls
	return &ls
}

// newValue allocates a new Value with the given fields and places it at the end of b.Values.
func (f *Func) newValue(op Op, t *types.Type, b *Block, pos src.XPos) *Value {
	var v *Value
	if f.freeValues != nil {
		v = f.freeValues
		f.freeValues = v.argstorage[0]
		v.argstorage[0] = nil
	} else {
		ID := f.vid.get()
		if int(ID) < len(f.Cache.values) {
			v = &f.Cache.values[ID]
			v.ID = ID
		} else {
			v = &Value{ID: ID}
		}
	}
	v.Op = op
	v.Type = t
	v.Block = b
	if notStmtBoundary(op) {
		pos = pos.WithNotStmt()
	}
	v.Pos = pos
	b.Values = append(b.Values, v)
	return v
}

// newValueNoBlock allocates a new Value with the given fields.
// The returned value is not placed in any block.  Once the caller
// decides on a block b, it must set b.Block and append
// the returned value to b.Values.
func (f *Func) newValueNoBlock(op Op, t *types.Type, pos src.XPos) *Value {
	var v *Value
	if f.freeValues != nil {
		v = f.freeValues
		f.freeValues = v.argstorage[0]
		v.argstorage[0] = nil
	} else {
		ID := f.vid.get()
		if int(ID) < len(f.Cache.values) {
			v = &f.Cache.values[ID]
			v.ID = ID
		} else {
			v = &Value{ID: ID}
		}
	}
	v.Op = op
	v.Type = t
	v.Block = nil // caller must fix this.
	if notStmtBoundary(op) {
		pos = pos.WithNotStmt()
	}
	v.Pos = pos
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
	f.Warnl(f.Entry.Pos, "\t%s\t%s%s\t%s", n, key, value, f.Name)
}

// unCacheLine removes v from f's constant cache "line" for aux,
// resets v.InCache when it is found (and removed),
// and returns whether v was found in that line.
func (f *Func) unCacheLine(v *Value, aux int64) bool {
	vv := f.constants[aux]
	for i, cv := range vv {
		if v == cv {
			vv[i] = vv[len(vv)-1]
			vv[len(vv)-1] = nil
			f.constants[aux] = vv[0 : len(vv)-1]
			v.InCache = false
			return true
		}
	}
	return false
}

// unCache removes v from f's constant cache.
func (f *Func) unCache(v *Value) {
	if v.InCache {
		aux := v.AuxInt
		if f.unCacheLine(v, aux) {
			return
		}
		if aux == 0 {
			switch v.Op {
			case OpConstNil:
				aux = constNilMagic
			case OpConstSlice:
				aux = constSliceMagic
			case OpConstString:
				aux = constEmptyStringMagic
			case OpConstInterface:
				aux = constInterfaceMagic
			}
			if aux != 0 && f.unCacheLine(v, aux) {
				return
			}
		}
		f.Fatalf("unCached value %s not found in cache, auxInt=0x%x, adjusted aux=0x%x", v.LongString(), v.AuxInt, aux)
	}
}

// freeValue frees a value. It must no longer be referenced or have any args.
func (f *Func) freeValue(v *Value) {
	if v.Block == nil {
		f.Fatalf("trying to free an already freed value")
	}
	if v.Uses != 0 {
		f.Fatalf("value %s still has %d uses", v, v.Uses)
	}
	if len(v.Args) != 0 {
		f.Fatalf("value %s still has %d args", v, len(v.Args))
	}
	// Clear everything but ID (which we reuse).
	id := v.ID
	if v.InCache {
		f.unCache(v)
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
		if int(ID) < len(f.Cache.blocks) {
			b = &f.Cache.blocks[ID]
			b.ID = ID
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
func (b *Block) NewValue0(pos src.XPos, op Op, t *types.Type) *Value {
	v := b.Func.newValue(op, t, b, pos)
	v.AuxInt = 0
	v.Args = v.argstorage[:0]
	return v
}

// NewValue returns a new value in the block with no arguments and an auxint value.
func (b *Block) NewValue0I(pos src.XPos, op Op, t *types.Type, auxint int64) *Value {
	v := b.Func.newValue(op, t, b, pos)
	v.AuxInt = auxint
	v.Args = v.argstorage[:0]
	return v
}

// NewValue returns a new value in the block with no arguments and an aux value.
func (b *Block) NewValue0A(pos src.XPos, op Op, t *types.Type, aux Aux) *Value {
	v := b.Func.newValue(op, t, b, pos)
	v.AuxInt = 0
	v.Aux = aux
	v.Args = v.argstorage[:0]
	return v
}

// NewValue returns a new value in the block with no arguments and both an auxint and aux values.
func (b *Block) NewValue0IA(pos src.XPos, op Op, t *types.Type, auxint int64, aux Aux) *Value {
	v := b.Func.newValue(op, t, b, pos)
	v.AuxInt = auxint
	v.Aux = aux
	v.Args = v.argstorage[:0]
	return v
}

// NewValue1 returns a new value in the block with one argument and zero aux values.
func (b *Block) NewValue1(pos src.XPos, op Op, t *types.Type, arg *Value) *Value {
	v := b.Func.newValue(op, t, b, pos)
	v.AuxInt = 0
	v.Args = v.argstorage[:1]
	v.argstorage[0] = arg
	arg.Uses++
	return v
}

// NewValue1I returns a new value in the block with one argument and an auxint value.
func (b *Block) NewValue1I(pos src.XPos, op Op, t *types.Type, auxint int64, arg *Value) *Value {
	v := b.Func.newValue(op, t, b, pos)
	v.AuxInt = auxint
	v.Args = v.argstorage[:1]
	v.argstorage[0] = arg
	arg.Uses++
	return v
}

// NewValue1A returns a new value in the block with one argument and an aux value.
func (b *Block) NewValue1A(pos src.XPos, op Op, t *types.Type, aux Aux, arg *Value) *Value {
	v := b.Func.newValue(op, t, b, pos)
	v.AuxInt = 0
	v.Aux = aux
	v.Args = v.argstorage[:1]
	v.argstorage[0] = arg
	arg.Uses++
	return v
}

// NewValue1IA returns a new value in the block with one argument and both an auxint and aux values.
func (b *Block) NewValue1IA(pos src.XPos, op Op, t *types.Type, auxint int64, aux Aux, arg *Value) *Value {
	v := b.Func.newValue(op, t, b, pos)
	v.AuxInt = auxint
	v.Aux = aux
	v.Args = v.argstorage[:1]
	v.argstorage[0] = arg
	arg.Uses++
	return v
}

// NewValue2 returns a new value in the block with two arguments and zero aux values.
func (b *Block) NewValue2(pos src.XPos, op Op, t *types.Type, arg0, arg1 *Value) *Value {
	v := b.Func.newValue(op, t, b, pos)
	v.AuxInt = 0
	v.Args = v.argstorage[:2]
	v.argstorage[0] = arg0
	v.argstorage[1] = arg1
	arg0.Uses++
	arg1.Uses++
	return v
}

// NewValue2A returns a new value in the block with two arguments and one aux values.
func (b *Block) NewValue2A(pos src.XPos, op Op, t *types.Type, aux Aux, arg0, arg1 *Value) *Value {
	v := b.Func.newValue(op, t, b, pos)
	v.AuxInt = 0
	v.Aux = aux
	v.Args = v.argstorage[:2]
	v.argstorage[0] = arg0
	v.argstorage[1] = arg1
	arg0.Uses++
	arg1.Uses++
	return v
}

// NewValue2I returns a new value in the block with two arguments and an auxint value.
func (b *Block) NewValue2I(pos src.XPos, op Op, t *types.Type, auxint int64, arg0, arg1 *Value) *Value {
	v := b.Func.newValue(op, t, b, pos)
	v.AuxInt = auxint
	v.Args = v.argstorage[:2]
	v.argstorage[0] = arg0
	v.argstorage[1] = arg1
	arg0.Uses++
	arg1.Uses++
	return v
}

// NewValue2IA returns a new value in the block with two arguments and both an auxint and aux values.
func (b *Block) NewValue2IA(pos src.XPos, op Op, t *types.Type, auxint int64, aux Aux, arg0, arg1 *Value) *Value {
	v := b.Func.newValue(op, t, b, pos)
	v.AuxInt = auxint
	v.Aux = aux
	v.Args = v.argstorage[:2]
	v.argstorage[0] = arg0
	v.argstorage[1] = arg1
	arg0.Uses++
	arg1.Uses++
	return v
}

// NewValue3 returns a new value in the block with three arguments and zero aux values.
func (b *Block) NewValue3(pos src.XPos, op Op, t *types.Type, arg0, arg1, arg2 *Value) *Value {
	v := b.Func.newValue(op, t, b, pos)
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
func (b *Block) NewValue3I(pos src.XPos, op Op, t *types.Type, auxint int64, arg0, arg1, arg2 *Value) *Value {
	v := b.Func.newValue(op, t, b, pos)
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

// NewValue3A returns a new value in the block with three argument and an aux value.
func (b *Block) NewValue3A(pos src.XPos, op Op, t *types.Type, aux Aux, arg0, arg1, arg2 *Value) *Value {
	v := b.Func.newValue(op, t, b, pos)
	v.AuxInt = 0
	v.Aux = aux
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
func (b *Block) NewValue4(pos src.XPos, op Op, t *types.Type, arg0, arg1, arg2, arg3 *Value) *Value {
	v := b.Func.newValue(op, t, b, pos)
	v.AuxInt = 0
	v.Args = []*Value{arg0, arg1, arg2, arg3}
	arg0.Uses++
	arg1.Uses++
	arg2.Uses++
	arg3.Uses++
	return v
}

// NewValue4I returns a new value in the block with four arguments and auxint value.
func (b *Block) NewValue4I(pos src.XPos, op Op, t *types.Type, auxint int64, arg0, arg1, arg2, arg3 *Value) *Value {
	v := b.Func.newValue(op, t, b, pos)
	v.AuxInt = auxint
	v.Args = []*Value{arg0, arg1, arg2, arg3}
	arg0.Uses++
	arg1.Uses++
	arg2.Uses++
	arg3.Uses++
	return v
}

// constVal returns a constant value for c.
func (f *Func) constVal(op Op, t *types.Type, c int64, setAuxInt bool) *Value {
	if f.constants == nil {
		f.constants = make(map[int64][]*Value)
	}
	vv := f.constants[c]
	for _, v := range vv {
		if v.Op == op && v.Type.Compare(t) == types.CMPeq {
			if setAuxInt && v.AuxInt != c {
				panic(fmt.Sprintf("cached const %s should have AuxInt of %d", v.LongString(), c))
			}
			return v
		}
	}
	var v *Value
	if setAuxInt {
		v = f.Entry.NewValue0I(src.NoXPos, op, t, c)
	} else {
		v = f.Entry.NewValue0(src.NoXPos, op, t)
	}
	f.constants[c] = append(vv, v)
	v.InCache = true
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
func (f *Func) ConstBool(t *types.Type, c bool) *Value {
	i := int64(0)
	if c {
		i = 1
	}
	return f.constVal(OpConstBool, t, i, true)
}
func (f *Func) ConstInt8(t *types.Type, c int8) *Value {
	return f.constVal(OpConst8, t, int64(c), true)
}
func (f *Func) ConstInt16(t *types.Type, c int16) *Value {
	return f.constVal(OpConst16, t, int64(c), true)
}
func (f *Func) ConstInt32(t *types.Type, c int32) *Value {
	return f.constVal(OpConst32, t, int64(c), true)
}
func (f *Func) ConstInt64(t *types.Type, c int64) *Value {
	return f.constVal(OpConst64, t, c, true)
}
func (f *Func) ConstFloat32(t *types.Type, c float64) *Value {
	return f.constVal(OpConst32F, t, int64(math.Float64bits(float64(float32(c)))), true)
}
func (f *Func) ConstFloat64(t *types.Type, c float64) *Value {
	return f.constVal(OpConst64F, t, int64(math.Float64bits(c)), true)
}

func (f *Func) ConstSlice(t *types.Type) *Value {
	return f.constVal(OpConstSlice, t, constSliceMagic, false)
}
func (f *Func) ConstInterface(t *types.Type) *Value {
	return f.constVal(OpConstInterface, t, constInterfaceMagic, false)
}
func (f *Func) ConstNil(t *types.Type) *Value {
	return f.constVal(OpConstNil, t, constNilMagic, false)
}
func (f *Func) ConstEmptyString(t *types.Type) *Value {
	v := f.constVal(OpConstString, t, constEmptyStringMagic, false)
	v.Aux = StringToAux("")
	return v
}
func (f *Func) ConstOffPtrSP(t *types.Type, c int64, sp *Value) *Value {
	v := f.constVal(OpOffPtr, t, c, true)
	if len(v.Args) == 0 {
		v.AddArg(sp)
	}
	return v

}

func (f *Func) Frontend() Frontend                                  { return f.fe }
func (f *Func) Warnl(pos src.XPos, msg string, args ...interface{}) { f.fe.Warnl(pos, msg, args...) }
func (f *Func) Logf(msg string, args ...interface{})                { f.fe.Logf(msg, args...) }
func (f *Func) Log() bool                                           { return f.fe.Log() }

func (f *Func) Fatalf(msg string, args ...interface{}) {
	stats := "crashed"
	if f.Log() {
		f.Logf("  pass %s end %s\n", f.pass.name, stats)
		printFunc(f)
	}
	if f.HTMLWriter != nil {
		f.HTMLWriter.WritePhase(f.pass.name, fmt.Sprintf("%s <span class=\"stats\">%s</span>", f.pass.name, stats))
		f.HTMLWriter.flushPhases()
	}
	f.fe.Fatalf(f.Entry.Pos, msg, args...)
}

// postorder returns the reachable blocks in f in a postorder traversal.
func (f *Func) postorder() []*Block {
	if f.cachedPostorder == nil {
		f.cachedPostorder = postorder(f)
	}
	return f.cachedPostorder
}

func (f *Func) Postorder() []*Block {
	return f.postorder()
}

// Idom returns a map from block ID to the immediate dominator of that block.
// f.Entry.ID maps to nil. Unreachable blocks map to nil as well.
func (f *Func) Idom() []*Block {
	if f.cachedIdom == nil {
		f.cachedIdom = dominators(f)
	}
	return f.cachedIdom
}

// Sdom returns a sparse tree representing the dominator relationships
// among the blocks of f.
func (f *Func) Sdom() SparseTree {
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

// DebugHashMatch reports whether environment variable evname
//  1. is empty (this is a special more-quickly implemented case of 3)
//  2. is "y" or "Y"
//  3. is a suffix of the sha1 hash of name
//  4. is a suffix of the environment variable
//     fmt.Sprintf("%s%d", evname, n)
//     provided that all such variables are nonempty for 0 <= i <= n
//
// Otherwise it returns false.
// When true is returned the message
//
//	"%s triggered %s\n", evname, name
//
// is printed on the file named in environment variable
//
//	GSHS_LOGFILE
//
// or standard out if that is empty or there is an error
// opening the file.
func (f *Func) DebugHashMatch(evname string) bool {
	name := f.fe.MyImportPath() + "." + f.Name
	evhash := os.Getenv(evname)
	switch evhash {
	case "":
		return true // default behavior with no EV is "on"
	case "y", "Y":
		f.logDebugHashMatch(evname, name)
		return true
	case "n", "N":
		return false
	}
	// Check the hash of the name against a partial input hash.
	// We use this feature to do a binary search to
	// find a function that is incorrectly compiled.
	hstr := ""
	for _, b := range sha1.Sum([]byte(name)) {
		hstr += fmt.Sprintf("%08b", b)
	}

	if strings.HasSuffix(hstr, evhash) {
		f.logDebugHashMatch(evname, name)
		return true
	}

	// Iteratively try additional hashes to allow tests for multi-point
	// failure.
	for i := 0; true; i++ {
		ev := fmt.Sprintf("%s%d", evname, i)
		evv := os.Getenv(ev)
		if evv == "" {
			break
		}
		if strings.HasSuffix(hstr, evv) {
			f.logDebugHashMatch(ev, name)
			return true
		}
	}
	return false
}

func (f *Func) logDebugHashMatch(evname, name string) {
	if f.logfiles == nil {
		f.logfiles = make(map[string]writeSyncer)
	}
	file := f.logfiles[evname]
	if file == nil {
		file = os.Stdout
		if tmpfile := os.Getenv("GSHS_LOGFILE"); tmpfile != "" {
			var err error
			file, err = os.OpenFile(tmpfile, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
			if err != nil {
				f.Fatalf("could not open hash-testing logfile %s", tmpfile)
			}
		}
		f.logfiles[evname] = file
	}
	fmt.Fprintf(file, "%s triggered %s\n", evname, name)
	file.Sync()
}

func DebugNameMatch(evname, name string) bool {
	return os.Getenv(evname) == name
}

func (f *Func) spSb() (sp, sb *Value) {
	initpos := src.NoXPos // These are originally created with no position in ssa.go; if they are optimized out then recreated, should be the same.
	for _, v := range f.Entry.Values {
		if v.Op == OpSB {
			sb = v
		}
		if v.Op == OpSP {
			sp = v
		}
		if sb != nil && sp != nil {
			return
		}
	}
	if sb == nil {
		sb = f.Entry.NewValue0(initpos.WithNotStmt(), OpSB, f.Config.Types.Uintptr)
	}
	if sp == nil {
		sp = f.Entry.NewValue0(initpos.WithNotStmt(), OpSP, f.Config.Types.Uintptr)
	}
	return
}
