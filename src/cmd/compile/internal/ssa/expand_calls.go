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

// expandCalls converts LE (Late Expansion) calls that act like they receive value args into a lower-level form
// that is more oriented to a platform's ABI.  The SelectN operations that extract results are rewritten into
// more appropriate forms, and any StructMake or ArrayMake inputs are decomposed until non-struct values are
// reached (for now, Strings, Slices, Complex, and Interface are not decomposed because they are rewritten in
// a subsequent phase, but that may need to change for a register ABI in case one of those composite values is
// split between registers and memory).
//
// TODO: when it comes time to use registers, might want to include builtin selectors as well, but currently that happens in lower.
func expandCalls(f *Func) {
	if !LateCallExpansionEnabledWithin(f) {
		return
	}
	canSSAType := f.fe.CanSSA
	regSize := f.Config.RegSize
	sp, _ := f.spSb()

	debug := f.pass.debug > 0

	// For 32-bit, need to deal with decomposition of 64-bit integers
	tUint32 := types.Types[types.TUINT32]
	tInt32 := types.Types[types.TINT32]
	var hiOffset, lowOffset int64
	if f.Config.BigEndian {
		lowOffset = 4
	} else {
		hiOffset = 4
	}

	pairTypes := func(et types.EType) (tHi, tLo *types.Type) {
		tHi = tUint32
		if et == types.TINT64 {
			tHi = tInt32
		}
		tLo = tUint32
		return
	}

	// isAlreadyExpandedAggregateType returns whether a type is an SSA-able "aggregate" (multiple register) type
	// that was expanded in an earlier phase (small user-defined arrays and structs, lowered in decomposeUser).
	// Other aggregate types are expanded in decomposeBuiltin, which comes later.
	isAlreadyExpandedAggregateType := func(t *types.Type) bool {
		if !canSSAType(t) {
			return false
		}
		return t.IsStruct() || t.IsArray() || regSize == 4 && t.Size() > 4 && t.IsInteger()
	}

	// Calls that need lowering have some number of inputs, including a memory input,
	// and produce a tuple of (value1, value2, ..., mem) where valueK may or may not be SSA-able.

	// With the current ABI those inputs need to be converted into stores to memory,
	// rethreading the call's memory input to the first, and the new call now receiving the last.

	// With the current ABI, the outputs need to be converted to loads, which will all use the call's
	// memory output as their input.

	// rewriteSelect recursively walks leaf selector to a root (OpSelectN) through
	// a chain of Struct/Array Select operations.  If the chain of selectors does not
	// end in OpSelectN, it does nothing (this can happen depending on compiler phase ordering).
	// It emits the code necessary to implement the leaf select operation that leads to the call.
	// TODO when registers really arrive, must also decompose anything split across two registers or registers and memory.
	var rewriteSelect func(leaf *Value, selector *Value, offset int64)
	rewriteSelect = func(leaf *Value, selector *Value, offset int64) {
		switch selector.Op {
		case OpSelectN:
			// TODO these may be duplicated. Should memoize. Intermediate selectors will go dead, no worries there.
			call := selector.Args[0]
			aux := call.Aux.(*AuxCall)
			which := selector.AuxInt
			if which == aux.NResults() { // mem is after the results.
				// rewrite v as a Copy of call -- the replacement call will produce a mem.
				leaf.copyOf(call)
			} else {
				leafType := leaf.Type
				pt := types.NewPtr(leafType)
				if canSSAType(leafType) {
					off := f.ConstOffPtrSP(pt, offset+aux.OffsetOfResult(which), sp)
					// Any selection right out of the arg area/registers has to be same Block as call, use call as mem input.
					if leaf.Block == call.Block {
						leaf.reset(OpLoad)
						leaf.SetArgs2(off, call)
					} else {
						w := call.Block.NewValue2(leaf.Pos, OpLoad, leafType, off, call)
						leaf.copyOf(w)
					}
				} else {
					panic("Should not have non-SSA-able OpSelectN")
				}
			}
		case OpStructSelect:
			w := selector.Args[0]
			if w.Type.Etype != types.TSTRUCT {
				fmt.Printf("Bad type for w:\nv=%v\nsel=%v\nw=%v\n,f=%s\n", leaf.LongString(), selector.LongString(), w.LongString(), f.Name)
			}
			rewriteSelect(leaf, w, offset+w.Type.FieldOff(int(selector.AuxInt)))

		case OpInt64Hi:
			w := selector.Args[0]
			rewriteSelect(leaf, w, offset+hiOffset)

		case OpInt64Lo:
			w := selector.Args[0]
			rewriteSelect(leaf, w, offset+lowOffset)

		case OpArraySelect:
			w := selector.Args[0]
			rewriteSelect(leaf, w, offset+selector.Type.Size()*selector.AuxInt)
		default:
			// Ignore dead ends; on 32-bit, these can occur running before decompose builtins.
		}
	}

	// storeArg converts stores of SSA-able aggregates into a series of stores of smaller types into
	// individual parameter slots.
	// TODO when registers really arrive, must also decompose anything split across two registers or registers and memory.
	var storeArg func(pos src.XPos, b *Block, a *Value, t *types.Type, offset int64, mem *Value) *Value
	storeArg = func(pos src.XPos, b *Block, a *Value, t *types.Type, offset int64, mem *Value) *Value {
		switch a.Op {
		case OpArrayMake0, OpStructMake0:
			return mem
		case OpStructMake1, OpStructMake2, OpStructMake3, OpStructMake4:
			for i := 0; i < t.NumFields(); i++ {
				fld := t.Field(i)
				mem = storeArg(pos, b, a.Args[i], fld.Type, offset+fld.Offset, mem)
			}
			return mem
		case OpArrayMake1:
			return storeArg(pos, b, a.Args[0], t.Elem(), offset, mem)

		case OpInt64Make:
			tHi, tLo := pairTypes(t.Etype)
			mem = storeArg(pos, b, a.Args[0], tHi, offset+hiOffset, mem)
			return storeArg(pos, b, a.Args[1], tLo, offset+lowOffset, mem)
		}
		dst := f.ConstOffPtrSP(types.NewPtr(t), offset, sp)
		x := b.NewValue3A(pos, OpStore, types.TypeMem, t, dst, a, mem)
		if debug {
			fmt.Printf("storeArg(%v) returns %s\n", a, x.LongString())
		}
		return x
	}

	// offsetFrom creates an offset from a pointer, simplifying chained offsets and offsets from SP
	// TODO should also optimize offsets from SB?
	offsetFrom := func(dst *Value, offset int64, t *types.Type) *Value {
		pt := types.NewPtr(t)
		if offset == 0 && dst.Type == pt { // this is not actually likely
			return dst
		}
		if dst.Op != OpOffPtr {
			return dst.Block.NewValue1I(dst.Pos.WithNotStmt(), OpOffPtr, pt, offset, dst)
		}
		// Simplify OpOffPtr
		from := dst.Args[0]
		offset += dst.AuxInt
		if from == sp {
			return f.ConstOffPtrSP(pt, offset, sp)
		}
		return dst.Block.NewValue1I(dst.Pos.WithNotStmt(), OpOffPtr, pt, offset, from)
	}

	// splitStore converts a store of an SSA-able aggregate into a series of smaller stores, emitting
	// appropriate Struct/Array Select operations (which will soon go dead) to obtain the parts.
	var splitStore func(dst, src, mem, v *Value, t *types.Type, offset int64, firstStorePos src.XPos) *Value
	splitStore = func(dst, src, mem, v *Value, t *types.Type, offset int64, firstStorePos src.XPos) *Value {
		// TODO might be worth commoning up duplicate selectors, but since they go dead, maybe no point.
		pos := v.Pos.WithNotStmt()
		switch t.Etype {
		case types.TINT64, types.TUINT64:
			if t.Width == regSize {
				break
			}
			tHi, tLo := pairTypes(t.Etype)
			sel := src.Block.NewValue1(pos, OpInt64Hi, tHi, src)
			mem = splitStore(dst, sel, mem, v, tHi, offset+hiOffset, firstStorePos)
			firstStorePos = firstStorePos.WithNotStmt()
			sel = src.Block.NewValue1(pos, OpInt64Lo, tLo, src)
			return splitStore(dst, sel, mem, v, tLo, offset+lowOffset, firstStorePos)

		case types.TARRAY:
			elt := t.Elem()
			for i := int64(0); i < t.NumElem(); i++ {
				sel := src.Block.NewValue1I(pos, OpArraySelect, elt, i, src)
				mem = splitStore(dst, sel, mem, v, elt, offset+i*elt.Width, firstStorePos)
				firstStorePos = firstStorePos.WithNotStmt()
			}
			return mem
		case types.TSTRUCT:
			if src.Op == OpIData && t.NumFields() == 1 && t.Field(0).Type.Width == t.Width && t.Width == regSize   {
				// This peculiar test deals with accesses to immediate interface data.
				// It works okay because everything is the same size.
				// Example code that triggers this can be found in go/constant/value.go, function ToComplex
				// v119 (+881) = IData <intVal> v6
				// v121 (+882) = StaticLECall <floatVal,mem> {AuxCall{"".itof([intVal,0])[floatVal,8]}} [16] v119 v1
				// This corresponds to the generic rewrite rule "(StructSelect [0] (IData x)) => (IData x)"
				// Guard against "struct{struct{*foo}}"
				for t.Etype == types.TSTRUCT && t.NumFields() == 1 {
					t = t.Field(0).Type
				}
				if t.Etype == types.TSTRUCT || t.Etype == types.TARRAY {
					f.Fatalf("Did not expect to find IDATA-immediate with non-trivial struct in it")
				}
				break // handle the leaf type.
			}
			for i := 0; i < t.NumFields(); i++ {
				fld := t.Field(i)
				sel := src.Block.NewValue1I(pos, OpStructSelect, fld.Type, int64(i), src)
				mem = splitStore(dst, sel, mem, v, fld.Type, offset+fld.Offset, firstStorePos)
				firstStorePos = firstStorePos.WithNotStmt()
			}
			return mem
		}
		// Default, including for aggregates whose single element exactly fills their container
		// TODO this will be a problem for cast interfaces containing floats when we move to registers.
		x := v.Block.NewValue3A(firstStorePos, OpStore, types.TypeMem, t, offsetFrom(dst, offset, t), src, mem)
		if debug {
			fmt.Printf("splitStore(%v, %v, %v, %v) returns %s\n", dst, src, mem, v, x.LongString())
		}
		return x
	}

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
				src := a.Args[0]
				dst := f.ConstOffPtrSP(src.Type, aux.OffsetOfArg(auxI), sp)
				if a.Uses == 1 {
					a.reset(OpMove)
					a.Pos = pos
					a.Type = types.TypeMem
					a.Aux = aux.TypeOfArg(auxI)
					a.AuxInt = aux.SizeOfArg(auxI)
					a.SetArgs3(dst, src, mem)
					mem = a
				} else {
					mem = a.Block.NewValue3A(pos, OpMove, types.TypeMem, aux.TypeOfArg(auxI), dst, src, mem)
					mem.AuxInt = aux.SizeOfArg(auxI)
				}
			} else {
				mem = storeArg(pos, v.Block, a, aux.TypeOfArg(auxI), aux.OffsetOfArg(auxI), mem)
			}
		}
		v.resetArgs()
		return mem
	}

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

	// Step 1: any stores of aggregates remaining are believed to be sourced from call results.
	// Decompose those stores into a series of smaller stores, adding selection ops as necessary.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op == OpStore {
				t := v.Aux.(*types.Type)
				if isAlreadyExpandedAggregateType(t) {
					dst, src, mem := v.Args[0], v.Args[1], v.Args[2]
					mem = splitStore(dst, src, mem, v, t, 0, v.Pos)
					v.copyOf(mem)
				}
			}
		}
	}

	val2Preds := make(map[*Value]int32) // Used to accumulate dependency graph of selection operations for topological ordering.

	// Step 2: accumulate selection operations for rewrite in topological order.
	// Any select-for-addressing applied to call results can be transformed directly.
	// TODO this is overkill; with the transformation of aggregate references into series of leaf references, it is only necessary to remember and recurse on the leaves.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			// Accumulate chains of selectors for processing in topological order
			switch v.Op {
			case OpStructSelect, OpArraySelect, OpInt64Hi, OpInt64Lo:
				w := v.Args[0]
				switch w.Op {
				case OpStructSelect, OpArraySelect, OpInt64Hi, OpInt64Lo, OpSelectN:
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
			case OpSelectNAddr:
				// Do these directly, there are no chains of selectors.
				call := v.Args[0]
				which := v.AuxInt
				aux := call.Aux.(*AuxCall)
				pt := v.Type
				off := f.ConstOffPtrSP(pt, aux.OffsetOfResult(which), sp)
				v.copyOf(off)
			}
		}
	}

	// Compilation must be deterministic
	var ordered []*Value
	less := func(i, j int) bool { return ordered[i].ID < ordered[j].ID }

	// Step 3: Rewrite in topological order.  All chains of selectors end up in same block as the call.
	for len(val2Preds) > 0 {
		ordered = ordered[:0]
		for v, n := range val2Preds {
			if n == 0 {
				ordered = append(ordered, v)
			}
		}
		sort.Slice(ordered, less)
		for _, v := range ordered {
			for {
				w := v.Args[0]
				if debug {
					fmt.Printf("About to rewrite %s, args[0]=%s\n", v.LongString(), w.LongString())
				}
				delete(val2Preds, v)
				rewriteSelect(v, v, 0)
				v = w
				n, ok := val2Preds[v]
				if !ok {
					break
				}
				if n != 1 {
					val2Preds[v] = n - 1
					break
				}
				// Loop on new v; val2Preds[v] == 1 will be deleted in that iteration, no need to store zero.
			}
		}
	}

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
