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
	if !LateCallExpansionEnabledWithin(f) {
		return
	}
	debug := f.pass.debug > 0

	if debug {
		fmt.Printf("\nexpandsCalls(%s)\n", f.Name)
	}

	canSSAType := f.fe.CanSSA
	regSize := f.Config.RegSize
	sp, _ := f.spSb()
	typ := &f.Config.Types
	ptrSize := f.Config.PtrSize

	// For 32-bit, need to deal with decomposition of 64-bit integers, which depends on endianness.
	var hiOffset, lowOffset int64
	if f.Config.BigEndian {
		lowOffset = 4
	} else {
		hiOffset = 4
	}

	namedSelects := make(map[*Value][]namedVal)

	sdom := f.Sdom()

	common := make(map[selKey]*Value)

	// intPairTypes returns the pair of 32-bit int types needed to encode a 64-bit integer type on a target
	// that has no 64-bit integer registers.
	intPairTypes := func(et types.EType) (tHi, tLo *types.Type) {
		tHi = typ.UInt32
		if et == types.TINT64 {
			tHi = typ.Int32
		}
		tLo = typ.UInt32
		return
	}

	// isAlreadyExpandedAggregateType returns whether a type is an SSA-able "aggregate" (multiple register) type
	// that was expanded in an earlier phase (currently, expand_calls is intended to run after decomposeBuiltin,
	// so this is all aggregate types -- small struct and array, complex, interface, string, slice, and 64-bit
	// integer on 32-bit).
	isAlreadyExpandedAggregateType := func(t *types.Type) bool {
		if !canSSAType(t) {
			return false
		}
		return t.IsStruct() || t.IsArray() || t.IsComplex() || t.IsInterface() || t.IsString() || t.IsSlice() ||
			t.Size() > regSize && t.IsInteger()
	}

	offsets := make(map[offsetKey]*Value)

	// offsetFrom creates an offset from a pointer, simplifying chained offsets and offsets from SP
	// TODO should also optimize offsets from SB?
	offsetFrom := func(from *Value, offset int64, pt *types.Type) *Value {
		if offset == 0 && from.Type == pt { // this is not actually likely
			return from
		}
		// Simplify, canonicalize
		for from.Op == OpOffPtr {
			offset += from.AuxInt
			from = from.Args[0]
		}
		if from == sp {
			return f.ConstOffPtrSP(pt, offset, sp)
		}
		key := offsetKey{from, offset, pt}
		v := offsets[key]
		if v != nil {
			return v
		}
		v = from.Block.NewValue1I(from.Pos.WithNotStmt(), OpOffPtr, pt, offset, from)
		offsets[key] = v
		return v
	}

	// splitSlots splits one "field" (specified by sfx, offset, and ty) out of the LocalSlots in ls and returns the new LocalSlots this generates.
	splitSlots := func(ls []LocalSlot, sfx string, offset int64, ty *types.Type) []LocalSlot {
		var locs []LocalSlot
		for i := range ls {
			locs = append(locs, f.fe.SplitSlot(&ls[i], sfx, offset, ty))
		}
		return locs
	}

	// removeTrivialWrapperTypes unwraps layers of
	// struct { singleField SomeType } and [1]SomeType
	// until a non-wrapper type is reached.  This is useful
	// for working with assignments to/from interface data
	// fields (either second operand to OpIMake or OpIData)
	// where the wrapping or type conversion can be elided
	// because of type conversions/assertions in source code
	// that do not appear in SSA.
	removeTrivialWrapperTypes := func(t *types.Type) *types.Type {
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
	var rewriteSelect func(leaf *Value, selector *Value, offset int64) []LocalSlot
	rewriteSelect = func(leaf *Value, selector *Value, offset int64) []LocalSlot {
		if debug {
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
			if !isAlreadyExpandedAggregateType(selector.Type) {
				if leafType == selector.Type { // OpIData leads us here, sometimes.
					leaf.copyOf(selector)
				} else {
					f.Fatalf("Unexpected OpArg type, selector=%s, leaf=%s\n", selector.LongString(), leaf.LongString())
				}
				if debug {
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
				if debug {
					fmt.Printf("\tnew %s\n", w.LongString())
				}
			}
			for _, s := range namedSelects[selector] {
				locs = append(locs, f.Names[s.locIndex])
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
				f.Fatalf("Unexpected Load as selector, leaf=%s, selector=%s\n", leaf.LongString(), selector.LongString())
			}
			leaf.copyOf(selector)
			for _, s := range namedSelects[selector] {
				locs = append(locs, f.Names[s.locIndex])
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
				if canSSAType(leafType) {
					pt := types.NewPtr(leafType)
					off := offsetFrom(sp, offset+aux.OffsetOfResult(which), pt)
					// Any selection right out of the arg area/registers has to be same Block as call, use call as mem input.
					if leaf.Block == call.Block {
						leaf.reset(OpLoad)
						leaf.SetArgs2(off, call)
						leaf.Type = leafType
					} else {
						w := call.Block.NewValue2(leaf.Pos, OpLoad, leafType, off, call)
						leaf.copyOf(w)
						if debug {
							fmt.Printf("\tnew %s\n", w.LongString())
						}
					}
					for _, s := range namedSelects[selector] {
						locs = append(locs, f.Names[s.locIndex])
					}
				} else {
					f.Fatalf("Should not have non-SSA-able OpSelectN, selector=%s", selector.LongString())
				}
			}

		case OpStructSelect:
			w := selector.Args[0]
			var ls []LocalSlot
			if w.Type.Etype != types.TSTRUCT { // IData artifact
				ls = rewriteSelect(leaf, w, offset)
			} else {
				ls = rewriteSelect(leaf, w, offset+w.Type.FieldOff(int(selector.AuxInt)))
				if w.Op != OpIData {
					for _, l := range ls {
						locs = append(locs, f.fe.SplitStruct(l, int(selector.AuxInt)))
					}
				}
			}

		case OpArraySelect:
			w := selector.Args[0]
			rewriteSelect(leaf, w, offset+selector.Type.Size()*selector.AuxInt)

		case OpInt64Hi:
			w := selector.Args[0]
			ls := rewriteSelect(leaf, w, offset+hiOffset)
			locs = splitSlots(ls, ".hi", hiOffset, leafType)

		case OpInt64Lo:
			w := selector.Args[0]
			ls := rewriteSelect(leaf, w, offset+lowOffset)
			locs = splitSlots(ls, ".lo", lowOffset, leafType)

		case OpStringPtr:
			ls := rewriteSelect(leaf, selector.Args[0], offset)
			locs = splitSlots(ls, ".ptr", 0, typ.BytePtr)

		case OpSlicePtr:
			w := selector.Args[0]
			ls := rewriteSelect(leaf, w, offset)
			locs = splitSlots(ls, ".ptr", 0, types.NewPtr(w.Type.Elem()))

		case OpITab:
			w := selector.Args[0]
			ls := rewriteSelect(leaf, w, offset)
			sfx := ".itab"
			if w.Type.IsEmptyInterface() {
				sfx = ".type"
			}
			locs = splitSlots(ls, sfx, 0, typ.Uintptr)

		case OpComplexReal:
			ls := rewriteSelect(leaf, selector.Args[0], offset)
			locs = splitSlots(ls, ".real", 0, leafType)

		case OpComplexImag:
			ls := rewriteSelect(leaf, selector.Args[0], offset+leafType.Width) // result is FloatNN, width of result is offset of imaginary part.
			locs = splitSlots(ls, ".imag", leafType.Width, leafType)

		case OpStringLen, OpSliceLen:
			ls := rewriteSelect(leaf, selector.Args[0], offset+ptrSize)
			locs = splitSlots(ls, ".len", ptrSize, leafType)

		case OpIData:
			ls := rewriteSelect(leaf, selector.Args[0], offset+ptrSize)
			locs = splitSlots(ls, ".data", ptrSize, leafType)

		case OpSliceCap:
			ls := rewriteSelect(leaf, selector.Args[0], offset+2*ptrSize)
			locs = splitSlots(ls, ".cap", 2*ptrSize, leafType)

		case OpCopy: // If it's an intermediate result, recurse
			locs = rewriteSelect(leaf, selector.Args[0], offset)
			for _, s := range namedSelects[selector] {
				// this copy may have had its own name, preserve that, too.
				locs = append(locs, f.Names[s.locIndex])
			}

		default:
			// Ignore dead ends. These can occur if this phase is run before decompose builtin (which is not intended, but allowed).
		}

		return locs
	}

	// storeArgOrLoad converts stores of SSA-able aggregate arguments (passed to a call) into a series of primitive-typed
	// stores of non-aggregate types.  It recursively walks up a chain of selectors until it reaches a Load or an Arg.
	// If it does not reach a Load or an Arg, nothing happens; this allows a little freedom in phase ordering.
	var storeArgOrLoad func(pos src.XPos, b *Block, base, source, mem *Value, t *types.Type, offset int64) *Value

	// decomposeArgOrLoad is a helper for storeArgOrLoad.
	// It decomposes a Load or an Arg into smaller parts, parameterized by the decomposeOne and decomposeTwo functions
	// passed to it, and returns the new mem. If the type does not match one of the expected aggregate types, it returns nil instead.
	decomposeArgOrLoad := func(pos src.XPos, b *Block, base, source, mem *Value, t *types.Type, offset int64,
		decomposeOne func(pos src.XPos, b *Block, base, source, mem *Value, t1 *types.Type, offArg, offStore int64) *Value,
		decomposeTwo func(pos src.XPos, b *Block, base, source, mem *Value, t1, t2 *types.Type, offArg, offStore int64) *Value) *Value {
		u := source.Type
		switch u.Etype {
		case types.TARRAY:
			elem := u.Elem()
			for i := int64(0); i < u.NumElem(); i++ {
				elemOff := i * elem.Size()
				mem = decomposeOne(pos, b, base, source, mem, elem, source.AuxInt+elemOff, offset+elemOff)
				pos = pos.WithNotStmt()
			}
			return mem
		case types.TSTRUCT:
			for i := 0; i < u.NumFields(); i++ {
				fld := u.Field(i)
				mem = decomposeOne(pos, b, base, source, mem, fld.Type, source.AuxInt+fld.Offset, offset+fld.Offset)
				pos = pos.WithNotStmt()
			}
			return mem
		case types.TINT64, types.TUINT64:
			if t.Width == regSize {
				break
			}
			tHi, tLo := intPairTypes(t.Etype)
			mem = decomposeOne(pos, b, base, source, mem, tHi, source.AuxInt+hiOffset, offset+hiOffset)
			pos = pos.WithNotStmt()
			return decomposeOne(pos, b, base, source, mem, tLo, source.AuxInt+lowOffset, offset+lowOffset)
		case types.TINTER:
			return decomposeTwo(pos, b, base, source, mem, typ.Uintptr, typ.BytePtr, source.AuxInt, offset)
		case types.TSTRING:
			return decomposeTwo(pos, b, base, source, mem, typ.BytePtr, typ.Int, source.AuxInt, offset)
		case types.TCOMPLEX64:
			return decomposeTwo(pos, b, base, source, mem, typ.Float32, typ.Float32, source.AuxInt, offset)
		case types.TCOMPLEX128:
			return decomposeTwo(pos, b, base, source, mem, typ.Float64, typ.Float64, source.AuxInt, offset)
		case types.TSLICE:
			mem = decomposeTwo(pos, b, base, source, mem, typ.BytePtr, typ.Int, source.AuxInt, offset)
			return decomposeOne(pos, b, base, source, mem, typ.Int, source.AuxInt+2*ptrSize, offset+2*ptrSize)
		}
		return nil
	}

	// storeOneArg creates a decomposed (one step) arg that is then stored.
	// pos and b locate the store instruction, base is the base of the store target, source is the "base" of the value input,
	// mem is the input mem, t is the type in question, and offArg and offStore are the offsets from the respective bases.
	storeOneArg := func(pos src.XPos, b *Block, base, source, mem *Value, t *types.Type, offArg, offStore int64) *Value {
		w := common[selKey{source, offArg, t.Width, t}]
		if w == nil {
			w = source.Block.NewValue0IA(source.Pos, OpArg, t, offArg, source.Aux)
			common[selKey{source, offArg, t.Width, t}] = w
		}
		return storeArgOrLoad(pos, b, base, w, mem, t, offStore)
	}

	// storeOneLoad creates a decomposed (one step) load that is then stored.
	storeOneLoad := func(pos src.XPos, b *Block, base, source, mem *Value, t *types.Type, offArg, offStore int64) *Value {
		from := offsetFrom(source.Args[0], offArg, types.NewPtr(t))
		w := source.Block.NewValue2(source.Pos, OpLoad, t, from, mem)
		return storeArgOrLoad(pos, b, base, w, mem, t, offStore)
	}

	storeTwoArg := func(pos src.XPos, b *Block, base, source, mem *Value, t1, t2 *types.Type, offArg, offStore int64) *Value {
		mem = storeOneArg(pos, b, base, source, mem, t1, offArg, offStore)
		pos = pos.WithNotStmt()
		t1Size := t1.Size()
		return storeOneArg(pos, b, base, source, mem, t2, offArg+t1Size, offStore+t1Size)
	}

	storeTwoLoad := func(pos src.XPos, b *Block, base, source, mem *Value, t1, t2 *types.Type, offArg, offStore int64) *Value {
		mem = storeOneLoad(pos, b, base, source, mem, t1, offArg, offStore)
		pos = pos.WithNotStmt()
		t1Size := t1.Size()
		return storeOneLoad(pos, b, base, source, mem, t2, offArg+t1Size, offStore+t1Size)
	}

	storeArgOrLoad = func(pos src.XPos, b *Block, base, source, mem *Value, t *types.Type, offset int64) *Value {
		if debug {
			fmt.Printf("\tstoreArgOrLoad(%s;  %s;  %s;  %s; %d)\n", base.LongString(), source.LongString(), mem.String(), t.String(), offset)
		}

		switch source.Op {
		case OpCopy:
			return storeArgOrLoad(pos, b, base, source.Args[0], mem, t, offset)

		case OpLoad:
			ret := decomposeArgOrLoad(pos, b, base, source, mem, t, offset, storeOneLoad, storeTwoLoad)
			if ret != nil {
				return ret
			}

		case OpArg:
			ret := decomposeArgOrLoad(pos, b, base, source, mem, t, offset, storeOneArg, storeTwoArg)
			if ret != nil {
				return ret
			}

		case OpArrayMake0, OpStructMake0:
			return mem

		case OpStructMake1, OpStructMake2, OpStructMake3, OpStructMake4:
			for i := 0; i < t.NumFields(); i++ {
				fld := t.Field(i)
				mem = storeArgOrLoad(pos, b, base, source.Args[i], mem, fld.Type, offset+fld.Offset)
				pos = pos.WithNotStmt()
			}
			return mem

		case OpArrayMake1:
			return storeArgOrLoad(pos, b, base, source.Args[0], mem, t.Elem(), offset)

		case OpInt64Make:
			tHi, tLo := intPairTypes(t.Etype)
			mem = storeArgOrLoad(pos, b, base, source.Args[0], mem, tHi, offset+hiOffset)
			pos = pos.WithNotStmt()
			return storeArgOrLoad(pos, b, base, source.Args[1], mem, tLo, offset+lowOffset)

		case OpComplexMake:
			tPart := typ.Float32
			wPart := t.Width / 2
			if wPart == 8 {
				tPart = typ.Float64
			}
			mem = storeArgOrLoad(pos, b, base, source.Args[0], mem, tPart, offset)
			pos = pos.WithNotStmt()
			return storeArgOrLoad(pos, b, base, source.Args[1], mem, tPart, offset+wPart)

		case OpIMake:
			mem = storeArgOrLoad(pos, b, base, source.Args[0], mem, typ.Uintptr, offset)
			pos = pos.WithNotStmt()
			return storeArgOrLoad(pos, b, base, source.Args[1], mem, typ.BytePtr, offset+ptrSize)

		case OpStringMake:
			mem = storeArgOrLoad(pos, b, base, source.Args[0], mem, typ.BytePtr, offset)
			pos = pos.WithNotStmt()
			return storeArgOrLoad(pos, b, base, source.Args[1], mem, typ.Int, offset+ptrSize)

		case OpSliceMake:
			mem = storeArgOrLoad(pos, b, base, source.Args[0], mem, typ.BytePtr, offset)
			pos = pos.WithNotStmt()
			mem = storeArgOrLoad(pos, b, base, source.Args[1], mem, typ.Int, offset+ptrSize)
			return storeArgOrLoad(pos, b, base, source.Args[2], mem, typ.Int, offset+2*ptrSize)
		}

		// For nodes that cannot be taken apart -- OpSelectN, other structure selectors.
		switch t.Etype {
		case types.TARRAY:
			elt := t.Elem()
			if source.Type != t && t.NumElem() == 1 && elt.Width == t.Width && t.Width == regSize {
				t = removeTrivialWrapperTypes(t)
				// it could be a leaf type, but the "leaf" could be complex64 (for example)
				return storeArgOrLoad(pos, b, base, source, mem, t, offset)
			}
			for i := int64(0); i < t.NumElem(); i++ {
				sel := source.Block.NewValue1I(pos, OpArraySelect, elt, i, source)
				mem = storeArgOrLoad(pos, b, base, sel, mem, elt, offset+i*elt.Width)
				pos = pos.WithNotStmt()
			}
			return mem

		case types.TSTRUCT:
			if source.Type != t && t.NumFields() == 1 && t.Field(0).Type.Width == t.Width && t.Width == regSize {
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
				return storeArgOrLoad(pos, b, base, source, mem, t, offset)
			}

			for i := 0; i < t.NumFields(); i++ {
				fld := t.Field(i)
				sel := source.Block.NewValue1I(pos, OpStructSelect, fld.Type, int64(i), source)
				mem = storeArgOrLoad(pos, b, base, sel, mem, fld.Type, offset+fld.Offset)
				pos = pos.WithNotStmt()
			}
			return mem

		case types.TINT64, types.TUINT64:
			if t.Width == regSize {
				break
			}
			tHi, tLo := intPairTypes(t.Etype)
			sel := source.Block.NewValue1(pos, OpInt64Hi, tHi, source)
			mem = storeArgOrLoad(pos, b, base, sel, mem, tHi, offset+hiOffset)
			pos = pos.WithNotStmt()
			sel = source.Block.NewValue1(pos, OpInt64Lo, tLo, source)
			return storeArgOrLoad(pos, b, base, sel, mem, tLo, offset+lowOffset)

		case types.TINTER:
			sel := source.Block.NewValue1(pos, OpITab, typ.BytePtr, source)
			mem = storeArgOrLoad(pos, b, base, sel, mem, typ.BytePtr, offset)
			pos = pos.WithNotStmt()
			sel = source.Block.NewValue1(pos, OpIData, typ.BytePtr, source)
			return storeArgOrLoad(pos, b, base, sel, mem, typ.BytePtr, offset+ptrSize)

		case types.TSTRING:
			sel := source.Block.NewValue1(pos, OpStringPtr, typ.BytePtr, source)
			mem = storeArgOrLoad(pos, b, base, sel, mem, typ.BytePtr, offset)
			pos = pos.WithNotStmt()
			sel = source.Block.NewValue1(pos, OpStringLen, typ.Int, source)
			return storeArgOrLoad(pos, b, base, sel, mem, typ.Int, offset+ptrSize)

		case types.TSLICE:
			et := types.NewPtr(t.Elem())
			sel := source.Block.NewValue1(pos, OpSlicePtr, et, source)
			mem = storeArgOrLoad(pos, b, base, sel, mem, et, offset)
			pos = pos.WithNotStmt()
			sel = source.Block.NewValue1(pos, OpSliceLen, typ.Int, source)
			mem = storeArgOrLoad(pos, b, base, sel, mem, typ.Int, offset+ptrSize)
			sel = source.Block.NewValue1(pos, OpSliceCap, typ.Int, source)
			return storeArgOrLoad(pos, b, base, sel, mem, typ.Int, offset+2*ptrSize)

		case types.TCOMPLEX64:
			sel := source.Block.NewValue1(pos, OpComplexReal, typ.Float32, source)
			mem = storeArgOrLoad(pos, b, base, sel, mem, typ.Float32, offset)
			pos = pos.WithNotStmt()
			sel = source.Block.NewValue1(pos, OpComplexImag, typ.Float32, source)
			return storeArgOrLoad(pos, b, base, sel, mem, typ.Float32, offset+4)

		case types.TCOMPLEX128:
			sel := source.Block.NewValue1(pos, OpComplexReal, typ.Float64, source)
			mem = storeArgOrLoad(pos, b, base, sel, mem, typ.Float64, offset)
			pos = pos.WithNotStmt()
			sel = source.Block.NewValue1(pos, OpComplexImag, typ.Float64, source)
			return storeArgOrLoad(pos, b, base, sel, mem, typ.Float64, offset+8)
		}

		dst := offsetFrom(base, offset, types.NewPtr(t))
		x := b.NewValue3A(pos, OpStore, types.TypeMem, t, dst, source, mem)
		if debug {
			fmt.Printf("\t\tstoreArg returns %s\n", x.LongString())
		}
		return x
	}

	// rewriteArgs removes all the Args from a call and converts the call args into appropriate
	// stores (or later, register movement).  Extra args for interface and closure calls are ignored,
	// but removed.
	rewriteArgs := func(v *Value, firstArg int) *Value {
		// Thread the stores on the memory arg
		aux := v.Aux.(*AuxCall)
		pos := v.Pos.WithNotStmt()
		m0 := v.Args[len(v.Args)-1]
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
					f.Fatalf("Op...LECall and OpDereference have mismatched mem, %s and %s", v.LongString(), a.LongString())
				}
				// "Dereference" of addressed (probably not-SSA-eligible) value becomes Move
				// TODO this will be more complicated with registers in the picture.
				source := a.Args[0]
				dst := f.ConstOffPtrSP(source.Type, aux.OffsetOfArg(auxI), sp)
				if a.Uses == 1 && a.Block == v.Block {
					a.reset(OpMove)
					a.Pos = pos
					a.Type = types.TypeMem
					a.Aux = aux.TypeOfArg(auxI)
					a.AuxInt = aux.SizeOfArg(auxI)
					a.SetArgs3(dst, source, mem)
					mem = a
				} else {
					mem = v.Block.NewValue3A(pos, OpMove, types.TypeMem, aux.TypeOfArg(auxI), dst, source, mem)
					mem.AuxInt = aux.SizeOfArg(auxI)
				}
			} else {
				if debug {
					fmt.Printf("storeArg %s, %v, %d\n", a.LongString(), aux.TypeOfArg(auxI), aux.OffsetOfArg(auxI))
				}
				mem = storeArgOrLoad(pos, v.Block, sp, a, mem, aux.TypeOfArg(auxI), aux.OffsetOfArg(auxI))
			}
		}
		v.resetArgs()
		return mem
	}

	// TODO if too slow, whole program iteration can be replaced w/ slices of appropriate values, accumulated in first loop here.

	// Step 0: rewrite the calls to convert incoming args to stores.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			switch v.Op {
			case OpStaticLECall:
				mem := rewriteArgs(v, 0)
				v.SetArgs1(mem)
			case OpClosureLECall:
				code := v.Args[0]
				context := v.Args[1]
				mem := rewriteArgs(v, 2)
				v.SetArgs3(code, context, mem)
			case OpInterLECall:
				code := v.Args[0]
				mem := rewriteArgs(v, 1)
				v.SetArgs2(code, mem)
			}
		}
	}

	for i, name := range f.Names {
		t := name.Type
		if isAlreadyExpandedAggregateType(t) {
			for j, v := range f.NamedValues[name] {
				if v.Op == OpSelectN || v.Op == OpArg && isAlreadyExpandedAggregateType(v.Type) {
					ns := namedSelects[v]
					namedSelects[v] = append(ns, namedVal{locIndex: i, valIndex: j})
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
				iAEATt := isAlreadyExpandedAggregateType(t)

				if !iAEATt {
					// guarding against store immediate struct into interface data field -- store type is *uint8
					// TODO can this happen recursively?
					iAEATt = isAlreadyExpandedAggregateType(tSrc)
					if iAEATt {
						t = tSrc
					}
				}
				if iAEATt {
					if debug {
						fmt.Printf("Splitting store %s\n", v.LongString())
					}
					dst, mem := v.Args[0], v.Args[2]
					mem = storeArgOrLoad(v.Pos, b, dst, source, mem, t, 0)
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
					if debug {
						fmt.Printf("v2p[%s] = %d\n", w.LongString(), val2Preds[w])
					}
				}
				fallthrough

			case OpSelectN:
				if _, ok := val2Preds[v]; !ok {
					val2Preds[v] = 0
					if debug {
						fmt.Printf("v2p[%s] = %d\n", v.LongString(), val2Preds[v])
					}
				}

			case OpArg:
				if !isAlreadyExpandedAggregateType(v.Type) {
					continue
				}
				if _, ok := val2Preds[v]; !ok {
					val2Preds[v] = 0
					if debug {
						fmt.Printf("v2p[%s] = %d\n", v.LongString(), val2Preds[v])
					}
				}

			case OpSelectNAddr:
				// Do these directly, there are no chains of selectors.
				call := v.Args[0]
				which := v.AuxInt
				aux := call.Aux.(*AuxCall)
				pt := v.Type
				off := offsetFrom(sp, aux.OffsetOfResult(which), pt)
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
		return sdom.domorder(bi) > sdom.domorder(bj) // reverse the order to put dominators last.
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

	common = make(map[selKey]*Value)
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
			if w.Type.Etype == types.TSTRUCT {
				offset = w.Type.FieldOff(int(v.AuxInt))
			} else { // Immediate interface data artifact, offset is zero.
				f.Fatalf("Expand calls interface data problem, func %s, v=%s, w=%s\n", f.Name, v.LongString(), w.LongString())
			}
		case OpArraySelect:
			offset = size * v.AuxInt
		case OpSelectN:
			offset = w.Aux.(*AuxCall).OffsetOfResult(v.AuxInt)
		case OpInt64Hi:
			offset = hiOffset
		case OpInt64Lo:
			offset = lowOffset
		case OpStringLen, OpSliceLen, OpIData:
			offset = ptrSize
		case OpSliceCap:
			offset = 2 * ptrSize
		case OpComplexImag:
			offset = size
		}
		sk := selKey{from: w, size: size, offset: offset, typ: typ}
		dupe := common[sk]
		if dupe == nil {
			common[sk] = v
		} else if sdom.IsAncestorEq(dupe.Block, v.Block) {
			v.copyOf(dupe)
		} else {
			// Because values are processed in dominator order, the old common[s] will never dominate after a miss is seen.
			// Installing the new value might match some future values.
			common[sk] = v
		}
	}

	// Indices of entries in f.Names that need to be deleted.
	var toDelete []namedVal

	// Rewrite selectors.
	for i, v := range allOrdered {
		if debug {
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
		locs := rewriteSelect(v, v, 0)
		// Install new names.
		if v.Type.IsMemory() {
			continue
		}
		// Leaf types may have debug locations
		if !isAlreadyExpandedAggregateType(v.Type) {
			for _, l := range locs {
				f.NamedValues[l] = append(f.NamedValues[l], v)
			}
			f.Names = append(f.Names, locs...)
			continue
		}
		// Not-leaf types that had debug locations need to lose them.
		if ns, ok := namedSelects[v]; ok {
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
