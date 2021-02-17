// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"fmt"
	"sort"
)

type selKey struct {
	from   *Value
	offset int64
	size   int64
	typ    *types.Type
}

type offsetKey struct {
	from   *Value
	offset int64
	pt     *types.Type
}

func isBlockMultiValueExit(b *Block) bool {
	return (b.Kind == BlockRet || b.Kind == BlockRetJmp) && len(b.Controls) > 0 && b.Controls[0].Op == OpMakeResult
}

// removeTrivialWrapperTypes unwraps layers of
// struct { singleField SomeType } and [1]SomeType
// until a non-wrapper type is reached.  This is useful
// for working with assignments to/from interface data
// fields (either second operand to OpIMake or OpIData)
// where the wrapping or type conversion can be elided
// because of type conversions/assertions in source code
// that do not appear in SSA.
func removeTrivialWrapperTypes(t *types.Type) *types.Type {
	for {
		if t.IsStruct() && t.NumFields() == 1 {
			t = t.Field(0).Type
			continue
		}
		if t.IsArray() && t.NumElem() == 1 {
			t = t.Elem()
			continue
		}
		break
	}
	return t
}

type expandState struct {
	f            *Func
	debug        bool
	canSSAType   func(*types.Type) bool
	regSize      int64
	sp           *Value
	typs         *Types
	ptrSize      int64
	hiOffset     int64
	lowOffset    int64
	namedSelects map[*Value][]namedVal
	sdom         SparseTree
	common       map[selKey]*Value
	offsets      map[offsetKey]*Value
}

// intPairTypes returns the pair of 32-bit int types needed to encode a 64-bit integer type on a target
// that has no 64-bit integer registers.
func (x *expandState) intPairTypes(et types.Kind) (tHi, tLo *types.Type) {
	tHi = x.typs.UInt32
	if et == types.TINT64 {
		tHi = x.typs.Int32
	}
	tLo = x.typs.UInt32
	return
}

// isAlreadyExpandedAggregateType returns whether a type is an SSA-able "aggregate" (multiple register) type
// that was expanded in an earlier phase (currently, expand_calls is intended to run after decomposeBuiltin,
// so this is all aggregate types -- small struct and array, complex, interface, string, slice, and 64-bit
// integer on 32-bit).
func (x *expandState) isAlreadyExpandedAggregateType(t *types.Type) bool {
	if !x.canSSAType(t) {
		return false
	}
	return t.IsStruct() || t.IsArray() || t.IsComplex() || t.IsInterface() || t.IsString() || t.IsSlice() ||
		t.Size() > x.regSize && t.IsInteger()
}

// offsetFrom creates an offset from a pointer, simplifying chained offsets and offsets from SP
// TODO should also optimize offsets from SB?
func (x *expandState) offsetFrom(from *Value, offset int64, pt *types.Type) *Value {
	if offset == 0 && from.Type == pt { // this is not actually likely
		return from
	}
	// Simplify, canonicalize
	for from.Op == OpOffPtr {
		offset += from.AuxInt
		from = from.Args[0]
	}
	if from == x.sp {
		return x.f.ConstOffPtrSP(pt, offset, x.sp)
	}
	key := offsetKey{from, offset, pt}
	v := x.offsets[key]
	if v != nil {
		return v
	}
	v = from.Block.NewValue1I(from.Pos.WithNotStmt(), OpOffPtr, pt, offset, from)
	x.offsets[key] = v
	return v
}

// splitSlots splits one "field" (specified by sfx, offset, and ty) out of the LocalSlots in ls and returns the new LocalSlots this generates.
func (x *expandState) splitSlots(ls []LocalSlot, sfx string, offset int64, ty *types.Type) []LocalSlot {
	var locs []LocalSlot
	for i := range ls {
		locs = append(locs, x.f.fe.SplitSlot(&ls[i], sfx, offset, ty))
	}
	return locs
}

// Calls that need lowering have some number of inputs, including a memory input,
// and produce a tuple of (value1, value2, ..., mem) where valueK may or may not be SSA-able.

// With the current ABI those inputs need to be converted into stores to memory,
// rethreading the call's memory input to the first, and the new call now receiving the last.

// With the current ABI, the outputs need to be converted to loads, which will all use the call's
// memory output as their input.

// rewriteSelect recursively walks from leaf selector to a root (OpSelectN, OpLoad, OpArg)
// through a chain of Struct/Array/builtin Select operations.  If the chain of selectors does not
// end in an expected root, it does nothing (this can happen depending on compiler phase ordering).
// The "leaf" provides the type, the root supplies the container, and the leaf-to-root path
// accumulates the offset.
// It emits the code necessary to implement the leaf select operation that leads to the root.
//
// TODO when registers really arrive, must also decompose anything split across two registers or registers and memory.
func (x *expandState) rewriteSelect(leaf *Value, selector *Value, offset int64) []LocalSlot {
	if x.debug {
		fmt.Printf("rewriteSelect(%s, %s, %d)\n", leaf.LongString(), selector.LongString(), offset)
	}
	var locs []LocalSlot
	leafType := leaf.Type
	if len(selector.Args) > 0 {
		w := selector.Args[0]
		if w.Op == OpCopy {
			for w.Op == OpCopy {
				w = w.Args[0]
			}
			selector.SetArg(0, w)
		}
	}
	switch selector.Op {
	case OpArg:
		if !x.isAlreadyExpandedAggregateType(selector.Type) {
			if leafType == selector.Type { // OpIData leads us here, sometimes.
				leaf.copyOf(selector)
			} else {
				x.f.Fatalf("Unexpected OpArg type, selector=%s, leaf=%s\n", selector.LongString(), leaf.LongString())
			}
			if x.debug {
				fmt.Printf("\tOpArg, break\n")
			}
			break
		}
		switch leaf.Op {
		case OpIData, OpStructSelect, OpArraySelect:
			leafType = removeTrivialWrapperTypes(leaf.Type)
		}
		aux := selector.Aux
		auxInt := selector.AuxInt + offset
		if leaf.Block == selector.Block {
			leaf.reset(OpArg)
			leaf.Aux = aux
			leaf.AuxInt = auxInt
			leaf.Type = leafType
		} else {
			w := selector.Block.NewValue0IA(leaf.Pos, OpArg, leafType, auxInt, aux)
			leaf.copyOf(w)
			if x.debug {
				fmt.Printf("\tnew %s\n", w.LongString())
			}
		}
		for _, s := range x.namedSelects[selector] {
			locs = append(locs, x.f.Names[s.locIndex])
		}

	case OpLoad: // We end up here because of IData of immediate structures.
		// Failure case:
		// (note the failure case is very rare; w/o this case, make.bash and run.bash both pass, as well as
		// the hard cases of building {syscall,math,math/cmplx,math/bits,go/constant} on ppc64le and mips-softfloat).
		//
		// GOSSAFUNC='(*dumper).dump' go build -gcflags=-l -tags=math_big_pure_go cmd/compile/internal/gc
		// cmd/compile/internal/gc/dump.go:136:14: internal compiler error: '(*dumper).dump': not lowered: v827, StructSelect PTR PTR
		// b2: ← b1
		// v20 (+142) = StaticLECall <interface {},mem> {AuxCall{reflect.Value.Interface([reflect.Value,0])[interface {},24]}} [40] v8 v1
		// v21 (142) = SelectN <mem> [1] v20
		// v22 (142) = SelectN <interface {}> [0] v20
		// b15: ← b8
		// v71 (+143) = IData <Nodes> v22 (v[Nodes])
		// v73 (+146) = StaticLECall <[]*Node,mem> {AuxCall{"".Nodes.Slice([Nodes,0])[[]*Node,8]}} [32] v71 v21
		//
		// translates (w/o the "case OpLoad:" above) to:
		//
		// b2: ← b1
		// v20 (+142) = StaticCall <mem> {AuxCall{reflect.Value.Interface([reflect.Value,0])[interface {},24]}} [40] v715
		// v23 (142) = Load <*uintptr> v19 v20
		// v823 (142) = IsNonNil <bool> v23
		// v67 (+143) = Load <*[]*Node> v880 v20
		// b15: ← b8
		// v827 (146) = StructSelect <*[]*Node> [0] v67
		// v846 (146) = Store <mem> {*[]*Node} v769 v827 v20
		// v73 (+146) = StaticCall <mem> {AuxCall{"".Nodes.Slice([Nodes,0])[[]*Node,8]}} [32] v846
		// i.e., the struct select is generated and remains in because it is not applied to an actual structure.
		// The OpLoad was created to load the single field of the IData
		// This case removes that StructSelect.
		if leafType != selector.Type {
			x.f.Fatalf("Unexpected Load as selector, leaf=%s, selector=%s\n", leaf.LongString(), selector.LongString())
		}
		leaf.copyOf(selector)
		for _, s := range x.namedSelects[selector] {
			locs = append(locs, x.f.Names[s.locIndex])
		}

	case OpSelectN:
		// TODO these may be duplicated. Should memoize. Intermediate selectors will go dead, no worries there.
		call := selector.Args[0]
		aux := call.Aux.(*AuxCall)
		which := selector.AuxInt
		if which == aux.NResults() { // mem is after the results.
			// rewrite v as a Copy of call -- the replacement call will produce a mem.
			leaf.copyOf(call)
		} else {
			leafType := removeTrivialWrapperTypes(leaf.Type)
			if x.canSSAType(leafType) {
				pt := types.NewPtr(leafType)
				off := x.offsetFrom(x.sp, offset+aux.OffsetOfResult(which), pt)
				// Any selection right out of the arg area/registers has to be same Block as call, use call as mem input.
				if leaf.Block == call.Block {
					leaf.reset(OpLoad)
					leaf.SetArgs2(off, call)
					leaf.Type = leafType
				} else {
					w := call.Block.NewValue2(leaf.Pos, OpLoad, leafType, off, call)
					leaf.copyOf(w)
					if x.debug {
						fmt.Printf("\tnew %s\n", w.LongString())
					}
				}
				for _, s := range x.namedSelects[selector] {
					locs = append(locs, x.f.Names[s.locIndex])
				}
			} else {
				x.f.Fatalf("Should not have non-SSA-able OpSelectN, selector=%s", selector.LongString())
			}
		}

	case OpStructSelect:
		w := selector.Args[0]
		var ls []LocalSlot
		if w.Type.Kind() != types.TSTRUCT { // IData artifact
			ls = x.rewriteSelect(leaf, w, offset)
		} else {
			ls = x.rewriteSelect(leaf, w, offset+w.Type.FieldOff(int(selector.AuxInt)))
			if w.Op != OpIData {
				for _, l := range ls {
					locs = append(locs, x.f.fe.SplitStruct(l, int(selector.AuxInt)))
				}
			}
		}

	case OpArraySelect:
		w := selector.Args[0]
		x.rewriteSelect(leaf, w, offset+selector.Type.Size()*selector.AuxInt)

	case OpInt64Hi:
		w := selector.Args[0]
		ls := x.rewriteSelect(leaf, w, offset+x.hiOffset)
		locs = x.splitSlots(ls, ".hi", x.hiOffset, leafType)

	case OpInt64Lo:
		w := selector.Args[0]
		ls := x.rewriteSelect(leaf, w, offset+x.lowOffset)
		locs = x.splitSlots(ls, ".lo", x.lowOffset, leafType)

	case OpStringPtr:
		ls := x.rewriteSelect(leaf, selector.Args[0], offset)
		locs = x.splitSlots(ls, ".ptr", 0, x.typs.BytePtr)

	case OpSlicePtr:
		w := selector.Args[0]
		ls := x.rewriteSelect(leaf, w, offset)
		locs = x.splitSlots(ls, ".ptr", 0, types.NewPtr(w.Type.Elem()))

	case OpITab:
		w := selector.Args[0]
		ls := x.rewriteSelect(leaf, w, offset)
		sfx := ".itab"
		if w.Type.IsEmptyInterface() {
			sfx = ".type"
		}
		locs = x.splitSlots(ls, sfx, 0, x.typs.Uintptr)

	case OpComplexReal:
		ls := x.rewriteSelect(leaf, selector.Args[0], offset)
		locs = x.splitSlots(ls, ".real", 0, leafType)

	case OpComplexImag:
		ls := x.rewriteSelect(leaf, selector.Args[0], offset+leafType.Width) // result is FloatNN, width of result is offset of imaginary part.
		locs = x.splitSlots(ls, ".imag", leafType.Width, leafType)

	case OpStringLen, OpSliceLen:
		ls := x.rewriteSelect(leaf, selector.Args[0], offset+x.ptrSize)
		locs = x.splitSlots(ls, ".len", x.ptrSize, leafType)

	case OpIData:
		ls := x.rewriteSelect(leaf, selector.Args[0], offset+x.ptrSize)
		locs = x.splitSlots(ls, ".data", x.ptrSize, leafType)

	case OpSliceCap:
		ls := x.rewriteSelect(leaf, selector.Args[0], offset+2*x.ptrSize)
		locs = x.splitSlots(ls, ".cap", 2*x.ptrSize, leafType)

	case OpCopy: // If it's an intermediate result, recurse
		locs = x.rewriteSelect(leaf, selector.Args[0], offset)
		for _, s := range x.namedSelects[selector] {
			// this copy may have had its own name, preserve that, too.
			locs = append(locs, x.f.Names[s.locIndex])
		}

	default:
		// Ignore dead ends. These can occur if this phase is run before decompose builtin (which is not intended, but allowed).
	}

	return locs
}

func (x *expandState) rewriteDereference(b *Block, base, a, mem *Value, offset, size int64, typ *types.Type, pos src.XPos) *Value {
	source := a.Args[0]
	dst := x.offsetFrom(base, offset, source.Type)
	if a.Uses == 1 && a.Block == b {
		a.reset(OpMove)
		a.Pos = pos
		a.Type = types.TypeMem
		a.Aux = typ
		a.AuxInt = size
		a.SetArgs3(dst, source, mem)
		mem = a
	} else {
		mem = b.NewValue3A(pos, OpMove, types.TypeMem, typ, dst, source, mem)
		mem.AuxInt = size
	}
	return mem
}

// decomposeArgOrLoad is a helper for storeArgOrLoad.
// It decomposes a Load or an Arg into smaller parts, parameterized by the decomposeOne and decomposeTwo functions
// passed to it, and returns the new mem. If the type does not match one of the expected aggregate types, it returns nil instead.
func (x *expandState) decomposeArgOrLoad(pos src.XPos, b *Block, base, source, mem *Value, t *types.Type, offset int64,
	decomposeOne func(x *expandState, pos src.XPos, b *Block, base, source, mem *Value, t1 *types.Type, offArg, offStore int64) *Value,
	decomposeTwo func(x *expandState, pos src.XPos, b *Block, base, source, mem *Value, t1, t2 *types.Type, offArg, offStore int64) *Value) *Value {
	u := source.Type
	switch u.Kind() {
	case types.TARRAY:
		elem := u.Elem()
		for i := int64(0); i < u.NumElem(); i++ {
			elemOff := i * elem.Size()
			mem = decomposeOne(x, pos, b, base, source, mem, elem, source.AuxInt+elemOff, offset+elemOff)
			pos = pos.WithNotStmt()
		}
		return mem
	case types.TSTRUCT:
		for i := 0; i < u.NumFields(); i++ {
			fld := u.Field(i)
			mem = decomposeOne(x, pos, b, base, source, mem, fld.Type, source.AuxInt+fld.Offset, offset+fld.Offset)
			pos = pos.WithNotStmt()
		}
		return mem
	case types.TINT64, types.TUINT64:
		if t.Width == x.regSize {
			break
		}
		tHi, tLo := x.intPairTypes(t.Kind())
		mem = decomposeOne(x, pos, b, base, source, mem, tHi, source.AuxInt+x.hiOffset, offset+x.hiOffset)
		pos = pos.WithNotStmt()
		return decomposeOne(x, pos, b, base, source, mem, tLo, source.AuxInt+x.lowOffset, offset+x.lowOffset)
	case types.TINTER:
		return decomposeTwo(x, pos, b, base, source, mem, x.typs.Uintptr, x.typs.BytePtr, source.AuxInt, offset)
	case types.TSTRING:
		return decomposeTwo(x, pos, b, base, source, mem, x.typs.BytePtr, x.typs.Int, source.AuxInt, offset)
	case types.TCOMPLEX64:
		return decomposeTwo(x, pos, b, base, source, mem, x.typs.Float32, x.typs.Float32, source.AuxInt, offset)
	case types.TCOMPLEX128:
		return decomposeTwo(x, pos, b, base, source, mem, x.typs.Float64, x.typs.Float64, source.AuxInt, offset)
	case types.TSLICE:
		mem = decomposeTwo(x, pos, b, base, source, mem, x.typs.BytePtr, x.typs.Int, source.AuxInt, offset)
		return decomposeOne(x, pos, b, base, source, mem, x.typs.Int, source.AuxInt+2*x.ptrSize, offset+2*x.ptrSize)
	}
	return nil
}

// storeOneArg creates a decomposed (one step) arg that is then stored.
// pos and b locate the store instruction, base is the base of the store target, source is the "base" of the value input,
// mem is the input mem, t is the type in question, and offArg and offStore are the offsets from the respective bases.
func storeOneArg(x *expandState, pos src.XPos, b *Block, base, source, mem *Value, t *types.Type, offArg, offStore int64) *Value {
	w := x.common[selKey{source, offArg, t.Width, t}]
	if w == nil {
		w = source.Block.NewValue0IA(source.Pos, OpArg, t, offArg, source.Aux)
		x.common[selKey{source, offArg, t.Width, t}] = w
	}
	return x.storeArgOrLoad(pos, b, base, w, mem, t, offStore)
}

// storeOneLoad creates a decomposed (one step) load that is then stored.
func storeOneLoad(x *expandState, pos src.XPos, b *Block, base, source, mem *Value, t *types.Type, offArg, offStore int64) *Value {
	from := x.offsetFrom(source.Args[0], offArg, types.NewPtr(t))
	w := source.Block.NewValue2(source.Pos, OpLoad, t, from, mem)
	return x.storeArgOrLoad(pos, b, base, w, mem, t, offStore)
}

func storeTwoArg(x *expandState, pos src.XPos, b *Block, base, source, mem *Value, t1, t2 *types.Type, offArg, offStore int64) *Value {
	mem = storeOneArg(x, pos, b, base, source, mem, t1, offArg, offStore)
	pos = pos.WithNotStmt()
	t1Size := t1.Size()
	return storeOneArg(x, pos, b, base, source, mem, t2, offArg+t1Size, offStore+t1Size)
}

func storeTwoLoad(x *expandState, pos src.XPos, b *Block, base, source, mem *Value, t1, t2 *types.Type, offArg, offStore int64) *Value {
	mem = storeOneLoad(x, pos, b, base, source, mem, t1, offArg, offStore)
	pos = pos.WithNotStmt()
	t1Size := t1.Size()
	return storeOneLoad(x, pos, b, base, source, mem, t2, offArg+t1Size, offStore+t1Size)
}

// storeArgOrLoad converts stores of SSA-able aggregate arguments (passed to a call) into a series of primitive-typed
// stores of non-aggregate types.  It recursively walks up a chain of selectors until it reaches a Load or an Arg.
// If it does not reach a Load or an Arg, nothing happens; this allows a little freedom in phase ordering.
func (x *expandState) storeArgOrLoad(pos src.XPos, b *Block, base, source, mem *Value, t *types.Type, offset int64) *Value {
	if x.debug {
		fmt.Printf("\tstoreArgOrLoad(%s;  %s;  %s;  %s; %d)\n", base.LongString(), source.LongString(), mem.String(), t.String(), offset)
	}

	switch source.Op {
	case OpCopy:
		return x.storeArgOrLoad(pos, b, base, source.Args[0], mem, t, offset)

	case OpLoad:
		ret := x.decomposeArgOrLoad(pos, b, base, source, mem, t, offset, storeOneLoad, storeTwoLoad)
		if ret != nil {
			return ret
		}

	case OpArg:
		ret := x.decomposeArgOrLoad(pos, b, base, source, mem, t, offset, storeOneArg, storeTwoArg)
		if ret != nil {
			return ret
		}

	case OpArrayMake0, OpStructMake0:
		return mem

	case OpStructMake1, OpStructMake2, OpStructMake3, OpStructMake4:
		for i := 0; i < t.NumFields(); i++ {
			fld := t.Field(i)
			mem = x.storeArgOrLoad(pos, b, base, source.Args[i], mem, fld.Type, offset+fld.Offset)
			pos = pos.WithNotStmt()
		}
		return mem

	case OpArrayMake1:
		return x.storeArgOrLoad(pos, b, base, source.Args[0], mem, t.Elem(), offset)

	case OpInt64Make:
		tHi, tLo := x.intPairTypes(t.Kind())
		mem = x.storeArgOrLoad(pos, b, base, source.Args[0], mem, tHi, offset+x.hiOffset)
		pos = pos.WithNotStmt()
		return x.storeArgOrLoad(pos, b, base, source.Args[1], mem, tLo, offset+x.lowOffset)

	case OpComplexMake:
		tPart := x.typs.Float32
		wPart := t.Width / 2
		if wPart == 8 {
			tPart = x.typs.Float64
		}
		mem = x.storeArgOrLoad(pos, b, base, source.Args[0], mem, tPart, offset)
		pos = pos.WithNotStmt()
		return x.storeArgOrLoad(pos, b, base, source.Args[1], mem, tPart, offset+wPart)

	case OpIMake:
		mem = x.storeArgOrLoad(pos, b, base, source.Args[0], mem, x.typs.Uintptr, offset)
		pos = pos.WithNotStmt()
		return x.storeArgOrLoad(pos, b, base, source.Args[1], mem, x.typs.BytePtr, offset+x.ptrSize)

	case OpStringMake:
		mem = x.storeArgOrLoad(pos, b, base, source.Args[0], mem, x.typs.BytePtr, offset)
		pos = pos.WithNotStmt()
		return x.storeArgOrLoad(pos, b, base, source.Args[1], mem, x.typs.Int, offset+x.ptrSize)

	case OpSliceMake:
		mem = x.storeArgOrLoad(pos, b, base, source.Args[0], mem, x.typs.BytePtr, offset)
		pos = pos.WithNotStmt()
		mem = x.storeArgOrLoad(pos, b, base, source.Args[1], mem, x.typs.Int, offset+x.ptrSize)
		return x.storeArgOrLoad(pos, b, base, source.Args[2], mem, x.typs.Int, offset+2*x.ptrSize)
	}

	// For nodes that cannot be taken apart -- OpSelectN, other structure selectors.
	switch t.Kind() {
	case types.TARRAY:
		elt := t.Elem()
		if source.Type != t && t.NumElem() == 1 && elt.Width == t.Width && t.Width == x.regSize {
			t = removeTrivialWrapperTypes(t)
			// it could be a leaf type, but the "leaf" could be complex64 (for example)
			return x.storeArgOrLoad(pos, b, base, source, mem, t, offset)
		}
		for i := int64(0); i < t.NumElem(); i++ {
			sel := source.Block.NewValue1I(pos, OpArraySelect, elt, i, source)
			mem = x.storeArgOrLoad(pos, b, base, sel, mem, elt, offset+i*elt.Width)
			pos = pos.WithNotStmt()
		}
		return mem

	case types.TSTRUCT:
		if source.Type != t && t.NumFields() == 1 && t.Field(0).Type.Width == t.Width && t.Width == x.regSize {
			// This peculiar test deals with accesses to immediate interface data.
			// It works okay because everything is the same size.
			// Example code that triggers this can be found in go/constant/value.go, function ToComplex
			// v119 (+881) = IData <intVal> v6
			// v121 (+882) = StaticLECall <floatVal,mem> {AuxCall{"".itof([intVal,0])[floatVal,8]}} [16] v119 v1
			// This corresponds to the generic rewrite rule "(StructSelect [0] (IData x)) => (IData x)"
			// Guard against "struct{struct{*foo}}"
			// Other rewriting phases create minor glitches when they transform IData, for instance the
			// interface-typed Arg "x" of ToFloat in go/constant/value.go
			//   v6 (858) = Arg <Value> {x} (x[Value], x[Value])
			// is rewritten by decomposeArgs into
			//   v141 (858) = Arg <uintptr> {x}
			//   v139 (858) = Arg <*uint8> {x} [8]
			// because of a type case clause on line 862 of go/constant/value.go
			//  	case intVal:
			//		   return itof(x)
			// v139 is later stored as an intVal == struct{val *big.Int} which naively requires the fields of
			// of a *uint8, which does not succeed.
			t = removeTrivialWrapperTypes(t)
			// it could be a leaf type, but the "leaf" could be complex64 (for example)
			return x.storeArgOrLoad(pos, b, base, source, mem, t, offset)
		}

		for i := 0; i < t.NumFields(); i++ {
			fld := t.Field(i)
			sel := source.Block.NewValue1I(pos, OpStructSelect, fld.Type, int64(i), source)
			mem = x.storeArgOrLoad(pos, b, base, sel, mem, fld.Type, offset+fld.Offset)
			pos = pos.WithNotStmt()
		}
		return mem

	case types.TINT64, types.TUINT64:
		if t.Width == x.regSize {
			break
		}
		tHi, tLo := x.intPairTypes(t.Kind())
		sel := source.Block.NewValue1(pos, OpInt64Hi, tHi, source)
		mem = x.storeArgOrLoad(pos, b, base, sel, mem, tHi, offset+x.hiOffset)
		pos = pos.WithNotStmt()
		sel = source.Block.NewValue1(pos, OpInt64Lo, tLo, source)
		return x.storeArgOrLoad(pos, b, base, sel, mem, tLo, offset+x.lowOffset)

	case types.TINTER:
		sel := source.Block.NewValue1(pos, OpITab, x.typs.BytePtr, source)
		mem = x.storeArgOrLoad(pos, b, base, sel, mem, x.typs.BytePtr, offset)
		pos = pos.WithNotStmt()
		sel = source.Block.NewValue1(pos, OpIData, x.typs.BytePtr, source)
		return x.storeArgOrLoad(pos, b, base, sel, mem, x.typs.BytePtr, offset+x.ptrSize)

	case types.TSTRING:
		sel := source.Block.NewValue1(pos, OpStringPtr, x.typs.BytePtr, source)
		mem = x.storeArgOrLoad(pos, b, base, sel, mem, x.typs.BytePtr, offset)
		pos = pos.WithNotStmt()
		sel = source.Block.NewValue1(pos, OpStringLen, x.typs.Int, source)
		return x.storeArgOrLoad(pos, b, base, sel, mem, x.typs.Int, offset+x.ptrSize)

	case types.TSLICE:
		et := types.NewPtr(t.Elem())
		sel := source.Block.NewValue1(pos, OpSlicePtr, et, source)
		mem = x.storeArgOrLoad(pos, b, base, sel, mem, et, offset)
		pos = pos.WithNotStmt()
		sel = source.Block.NewValue1(pos, OpSliceLen, x.typs.Int, source)
		mem = x.storeArgOrLoad(pos, b, base, sel, mem, x.typs.Int, offset+x.ptrSize)
		sel = source.Block.NewValue1(pos, OpSliceCap, x.typs.Int, source)
		return x.storeArgOrLoad(pos, b, base, sel, mem, x.typs.Int, offset+2*x.ptrSize)

	case types.TCOMPLEX64:
		sel := source.Block.NewValue1(pos, OpComplexReal, x.typs.Float32, source)
		mem = x.storeArgOrLoad(pos, b, base, sel, mem, x.typs.Float32, offset)
		pos = pos.WithNotStmt()
		sel = source.Block.NewValue1(pos, OpComplexImag, x.typs.Float32, source)
		return x.storeArgOrLoad(pos, b, base, sel, mem, x.typs.Float32, offset+4)

	case types.TCOMPLEX128:
		sel := source.Block.NewValue1(pos, OpComplexReal, x.typs.Float64, source)
		mem = x.storeArgOrLoad(pos, b, base, sel, mem, x.typs.Float64, offset)
		pos = pos.WithNotStmt()
		sel = source.Block.NewValue1(pos, OpComplexImag, x.typs.Float64, source)
		return x.storeArgOrLoad(pos, b, base, sel, mem, x.typs.Float64, offset+8)
	}

	dst := x.offsetFrom(base, offset, types.NewPtr(t))
	s := b.NewValue3A(pos, OpStore, types.TypeMem, t, dst, source, mem)
	if x.debug {
		fmt.Printf("\t\tstoreArg returns %s\n", s.LongString())
	}
	return s
}

// rewriteArgs removes all the Args from a call and converts the call args into appropriate
// stores (or later, register movement).  Extra args for interface and closure calls are ignored,
// but removed.
func (x *expandState) rewriteArgs(v *Value, firstArg int) *Value {
	// Thread the stores on the memory arg
	aux := v.Aux.(*AuxCall)
	pos := v.Pos.WithNotStmt()
	m0 := v.MemoryArg()
	mem := m0
	for i, a := range v.Args {
		if i < firstArg {
			continue
		}
		if a == m0 { // mem is last.
			break
		}
		auxI := int64(i - firstArg)
		if a.Op == OpDereference {
			if a.MemoryArg() != m0 {
				x.f.Fatalf("Op...LECall and OpDereference have mismatched mem, %s and %s", v.LongString(), a.LongString())
			}
			// "Dereference" of addressed (probably not-SSA-eligible) value becomes Move
			// TODO this will be more complicated with registers in the picture.
			mem = x.rewriteDereference(v.Block, x.sp, a, mem, aux.OffsetOfArg(auxI), aux.SizeOfArg(auxI), aux.TypeOfArg(auxI), pos)
		} else {
			if x.debug {
				fmt.Printf("storeArg %s, %v, %d\n", a.LongString(), aux.TypeOfArg(auxI), aux.OffsetOfArg(auxI))
			}
			mem = x.storeArgOrLoad(pos, v.Block, x.sp, a, mem, aux.TypeOfArg(auxI), aux.OffsetOfArg(auxI))
		}
	}
	v.resetArgs()
	return mem
}

// expandCalls converts LE (Late Expansion) calls that act like they receive value args into a lower-level form
// that is more oriented to a platform's ABI.  The SelectN operations that extract results are rewritten into
// more appropriate forms, and any StructMake or ArrayMake inputs are decomposed until non-struct values are
// reached.  On the callee side, OpArg nodes are not decomposed until this phase is run.
// TODO results should not be lowered until this phase.
func expandCalls(f *Func) {
	// Calls that need lowering have some number of inputs, including a memory input,
	// and produce a tuple of (value1, value2, ..., mem) where valueK may or may not be SSA-able.

	// With the current ABI those inputs need to be converted into stores to memory,
	// rethreading the call's memory input to the first, and the new call now receiving the last.

	// With the current ABI, the outputs need to be converted to loads, which will all use the call's
	// memory output as their input.
	sp, _ := f.spSb()
	x := &expandState{
		f:            f,
		debug:        f.pass.debug > 0,
		canSSAType:   f.fe.CanSSA,
		regSize:      f.Config.RegSize,
		sp:           sp,
		typs:         &f.Config.Types,
		ptrSize:      f.Config.PtrSize,
		namedSelects: make(map[*Value][]namedVal),
		sdom:         f.Sdom(),
		common:       make(map[selKey]*Value),
		offsets:      make(map[offsetKey]*Value),
	}

	// For 32-bit, need to deal with decomposition of 64-bit integers, which depends on endianness.
	if f.Config.BigEndian {
		x.lowOffset = 4
	} else {
		x.hiOffset = 4
	}

	if x.debug {
		fmt.Printf("\nexpandsCalls(%s)\n", f.Name)
	}

	// TODO if too slow, whole program iteration can be replaced w/ slices of appropriate values, accumulated in first loop here.

	// Step 0: rewrite the calls to convert incoming args to stores.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			switch v.Op {
			case OpStaticLECall:
				mem := x.rewriteArgs(v, 0)
				v.SetArgs1(mem)
			case OpClosureLECall:
				code := v.Args[0]
				context := v.Args[1]
				mem := x.rewriteArgs(v, 2)
				v.SetArgs3(code, context, mem)
			case OpInterLECall:
				code := v.Args[0]
				mem := x.rewriteArgs(v, 1)
				v.SetArgs2(code, mem)
			}
		}
		if isBlockMultiValueExit(b) {
			// Very similar to code in rewriteArgs, but results instead of args.
			v := b.Controls[0]
			m0 := v.MemoryArg()
			mem := m0
			aux := f.OwnAux
			pos := v.Pos.WithNotStmt()
			for j, a := range v.Args {
				i := int64(j)
				if a == m0 {
					break
				}
				auxType := aux.TypeOfResult(i)
				auxBase := b.NewValue2A(v.Pos, OpLocalAddr, types.NewPtr(auxType), aux.results[i].Name, x.sp, mem)
				auxOffset := int64(0)
				auxSize := aux.SizeOfResult(i)
				if a.Op == OpDereference {
					// Avoid a self-move, and if one is detected try to remove the already-inserted VarDef for the assignment that won't happen.
					if dAddr, dMem := a.Args[0], a.Args[1]; dAddr.Op == OpLocalAddr && dAddr.Args[0].Op == OpSP &&
						dAddr.Args[1] == dMem && dAddr.Aux == aux.results[i].Name {
						if dMem.Op == OpVarDef && dMem.Aux == dAddr.Aux {
							dMem.copyOf(dMem.MemoryArg()) // elide the VarDef
						}
						continue
					}
					mem = x.rewriteDereference(v.Block, auxBase, a, mem, auxOffset, auxSize, auxType, pos)
				} else {
					if a.Op == OpLoad && a.Args[0].Op == OpLocalAddr {
						addr := a.Args[0]
						if addr.MemoryArg() == a.MemoryArg() && addr.Aux == aux.results[i].Name {
							continue
						}
					}
					mem = x.storeArgOrLoad(v.Pos, b, auxBase, a, mem, aux.TypeOfResult(i), auxOffset)
				}
			}
			b.SetControl(mem)
			v.reset(OpInvalid) // otherwise it can have a mem operand which will fail check(), even though it is dead.
		}
	}

	for i, name := range f.Names {
		t := name.Type
		if x.isAlreadyExpandedAggregateType(t) {
			for j, v := range f.NamedValues[name] {
				if v.Op == OpSelectN || v.Op == OpArg && x.isAlreadyExpandedAggregateType(v.Type) {
					ns := x.namedSelects[v]
					x.namedSelects[v] = append(ns, namedVal{locIndex: i, valIndex: j})
				}
			}
		}
	}

	// Step 1: any stores of aggregates remaining are believed to be sourced from call results or args.
	// Decompose those stores into a series of smaller stores, adding selection ops as necessary.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op == OpStore {
				t := v.Aux.(*types.Type)
				source := v.Args[1]
				tSrc := source.Type
				iAEATt := x.isAlreadyExpandedAggregateType(t)

				if !iAEATt {
					// guarding against store immediate struct into interface data field -- store type is *uint8
					// TODO can this happen recursively?
					iAEATt = x.isAlreadyExpandedAggregateType(tSrc)
					if iAEATt {
						t = tSrc
					}
				}
				if iAEATt {
					if x.debug {
						fmt.Printf("Splitting store %s\n", v.LongString())
					}
					dst, mem := v.Args[0], v.Args[2]
					mem = x.storeArgOrLoad(v.Pos, b, dst, source, mem, t, 0)
					v.copyOf(mem)
				}
			}
		}
	}

	val2Preds := make(map[*Value]int32) // Used to accumulate dependency graph of selection operations for topological ordering.

	// Step 2: transform or accumulate selection operations for rewrite in topological order.
	//
	// Aggregate types that have already (in earlier phases) been transformed must be lowered comprehensively to finish
	// the transformation (user-defined structs and arrays, slices, strings, interfaces, complex, 64-bit on 32-bit architectures),
	//
	// Any select-for-addressing applied to call results can be transformed directly.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			// Accumulate chains of selectors for processing in topological order
			switch v.Op {
			case OpStructSelect, OpArraySelect,
				OpIData, OpITab,
				OpStringPtr, OpStringLen,
				OpSlicePtr, OpSliceLen, OpSliceCap,
				OpComplexReal, OpComplexImag,
				OpInt64Hi, OpInt64Lo:
				w := v.Args[0]
				switch w.Op {
				case OpStructSelect, OpArraySelect, OpSelectN, OpArg:
					val2Preds[w] += 1
					if x.debug {
						fmt.Printf("v2p[%s] = %d\n", w.LongString(), val2Preds[w])
					}
				}
				fallthrough

			case OpSelectN:
				if _, ok := val2Preds[v]; !ok {
					val2Preds[v] = 0
					if x.debug {
						fmt.Printf("v2p[%s] = %d\n", v.LongString(), val2Preds[v])
					}
				}

			case OpArg:
				if !x.isAlreadyExpandedAggregateType(v.Type) {
					continue
				}
				if _, ok := val2Preds[v]; !ok {
					val2Preds[v] = 0
					if x.debug {
						fmt.Printf("v2p[%s] = %d\n", v.LongString(), val2Preds[v])
					}
				}

			case OpSelectNAddr:
				// Do these directly, there are no chains of selectors.
				call := v.Args[0]
				which := v.AuxInt
				aux := call.Aux.(*AuxCall)
				pt := v.Type
				off := x.offsetFrom(x.sp, aux.OffsetOfResult(which), pt)
				v.copyOf(off)
			}
		}
	}

	// Step 3: Compute topological order of selectors,
	// then process it in reverse to eliminate duplicates,
	// then forwards to rewrite selectors.
	//
	// All chains of selectors end up in same block as the call.

	// Compilation must be deterministic, so sort after extracting first zeroes from map.
	// Sorting allows dominators-last order within each batch,
	// so that the backwards scan for duplicates will most often find copies from dominating blocks (it is best-effort).
	var toProcess []*Value
	less := func(i, j int) bool {
		vi, vj := toProcess[i], toProcess[j]
		bi, bj := vi.Block, vj.Block
		if bi == bj {
			return vi.ID < vj.ID
		}
		return x.sdom.domorder(bi) > x.sdom.domorder(bj) // reverse the order to put dominators last.
	}

	// Accumulate order in allOrdered
	var allOrdered []*Value
	for v, n := range val2Preds {
		if n == 0 {
			allOrdered = append(allOrdered, v)
		}
	}
	last := 0 // allOrdered[0:last] has been top-sorted and processed
	for len(val2Preds) > 0 {
		toProcess = allOrdered[last:]
		last = len(allOrdered)
		sort.SliceStable(toProcess, less)
		for _, v := range toProcess {
			delete(val2Preds, v)
			if v.Op == OpArg {
				continue // no Args[0], hence done.
			}
			w := v.Args[0]
			n, ok := val2Preds[w]
			if !ok {
				continue
			}
			if n == 1 {
				allOrdered = append(allOrdered, w)
				delete(val2Preds, w)
				continue
			}
			val2Preds[w] = n - 1
		}
	}

	x.common = make(map[selKey]*Value)
	// Rewrite duplicate selectors as copies where possible.
	for i := len(allOrdered) - 1; i >= 0; i-- {
		v := allOrdered[i]
		if v.Op == OpArg {
			continue
		}
		w := v.Args[0]
		if w.Op == OpCopy {
			for w.Op == OpCopy {
				w = w.Args[0]
			}
			v.SetArg(0, w)
		}
		typ := v.Type
		if typ.IsMemory() {
			continue // handled elsewhere, not an indexable result
		}
		size := typ.Width
		offset := int64(0)
		switch v.Op {
		case OpStructSelect:
			if w.Type.Kind() == types.TSTRUCT {
				offset = w.Type.FieldOff(int(v.AuxInt))
			} else { // Immediate interface data artifact, offset is zero.
				f.Fatalf("Expand calls interface data problem, func %s, v=%s, w=%s\n", f.Name, v.LongString(), w.LongString())
			}
		case OpArraySelect:
			offset = size * v.AuxInt
		case OpSelectN:
			offset = w.Aux.(*AuxCall).OffsetOfResult(v.AuxInt)
		case OpInt64Hi:
			offset = x.hiOffset
		case OpInt64Lo:
			offset = x.lowOffset
		case OpStringLen, OpSliceLen, OpIData:
			offset = x.ptrSize
		case OpSliceCap:
			offset = 2 * x.ptrSize
		case OpComplexImag:
			offset = size
		}
		sk := selKey{from: w, size: size, offset: offset, typ: typ}
		dupe := x.common[sk]
		if dupe == nil {
			x.common[sk] = v
		} else if x.sdom.IsAncestorEq(dupe.Block, v.Block) {
			v.copyOf(dupe)
		} else {
			// Because values are processed in dominator order, the old common[s] will never dominate after a miss is seen.
			// Installing the new value might match some future values.
			x.common[sk] = v
		}
	}

	// Indices of entries in f.Names that need to be deleted.
	var toDelete []namedVal

	// Rewrite selectors.
	for i, v := range allOrdered {
		if x.debug {
			b := v.Block
			fmt.Printf("allOrdered[%d] = b%d, %s, uses=%d\n", i, b.ID, v.LongString(), v.Uses)
		}
		if v.Uses == 0 {
			v.reset(OpInvalid)
			continue
		}
		if v.Op == OpCopy {
			continue
		}
		locs := x.rewriteSelect(v, v, 0)
		// Install new names.
		if v.Type.IsMemory() {
			continue
		}
		// Leaf types may have debug locations
		if !x.isAlreadyExpandedAggregateType(v.Type) {
			for _, l := range locs {
				f.NamedValues[l] = append(f.NamedValues[l], v)
			}
			f.Names = append(f.Names, locs...)
			continue
		}
		// Not-leaf types that had debug locations need to lose them.
		if ns, ok := x.namedSelects[v]; ok {
			toDelete = append(toDelete, ns...)
		}
	}

	deleteNamedVals(f, toDelete)

	// Step 4: rewrite the calls themselves, correcting the type
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			switch v.Op {
			case OpStaticLECall:
				v.Op = OpStaticCall
				v.Type = types.TypeMem
			case OpClosureLECall:
				v.Op = OpClosureCall
				v.Type = types.TypeMem
			case OpInterLECall:
				v.Op = OpInterCall
				v.Type = types.TypeMem
			}
		}
	}

	// Step 5: elide any copies introduced.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			for i, a := range v.Args {
				if a.Op != OpCopy {
					continue
				}
				aa := copySource(a)
				v.SetArg(i, aa)
				for a.Uses == 0 {
					b := a.Args[0]
					a.reset(OpInvalid)
					a = b
				}
			}
		}
	}
}
