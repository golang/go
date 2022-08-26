// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/abi"
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"fmt"
	"sort"
)

type selKey struct {
	from          *Value // what is selected from
	offsetOrIndex int64  // whatever is appropriate for the selector
	size          int64
	typ           *types.Type
}

type Abi1RO uint8 // An offset within a parameter's slice of register indices, for abi1.

func isBlockMultiValueExit(b *Block) bool {
	return (b.Kind == BlockRet || b.Kind == BlockRetJmp) && b.Controls[0] != nil && b.Controls[0].Op == OpMakeResult
}

func badVal(s string, v *Value) error {
	return fmt.Errorf("%s %s", s, v.LongString())
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

// A registerCursor tracks which register is used for an Arg or regValues, or a piece of such.
type registerCursor struct {
	// TODO(register args) convert this to a generalized target cursor.
	storeDest *Value // if there are no register targets, then this is the base of the store.
	regsLen   int    // the number of registers available for this Arg/result (which is all in registers or not at all)
	nextSlice Abi1RO // the next register/register-slice offset
	config    *abi.ABIConfig
	regValues *[]*Value // values assigned to registers accumulate here
}

func (rc *registerCursor) String() string {
	dest := "<none>"
	if rc.storeDest != nil {
		dest = rc.storeDest.String()
	}
	regs := "<none>"
	if rc.regValues != nil {
		regs = ""
		for i, x := range *rc.regValues {
			if i > 0 {
				regs = regs + "; "
			}
			regs = regs + x.LongString()
		}
	}
	// not printing the config because that has not been useful
	return fmt.Sprintf("RCSR{storeDest=%v, regsLen=%d, nextSlice=%d, regValues=[%s]}", dest, rc.regsLen, rc.nextSlice, regs)
}

// next effectively post-increments the register cursor; the receiver is advanced,
// the old value is returned.
func (c *registerCursor) next(t *types.Type) registerCursor {
	rc := *c
	if int(c.nextSlice) < c.regsLen {
		w := c.config.NumParamRegs(t)
		c.nextSlice += Abi1RO(w)
	}
	return rc
}

// plus returns a register cursor offset from the original, without modifying the original.
func (c *registerCursor) plus(regWidth Abi1RO) registerCursor {
	rc := *c
	rc.nextSlice += regWidth
	return rc
}

const (
	// Register offsets for fields of built-in aggregate types; the ones not listed are zero.
	RO_complex_imag = 1
	RO_string_len   = 1
	RO_slice_len    = 1
	RO_slice_cap    = 2
	RO_iface_data   = 1
)

func (x *expandState) regWidth(t *types.Type) Abi1RO {
	return Abi1RO(x.abi1.NumParamRegs(t))
}

// regOffset returns the register offset of the i'th element of type t
func (x *expandState) regOffset(t *types.Type, i int) Abi1RO {
	// TODO maybe cache this in a map if profiling recommends.
	if i == 0 {
		return 0
	}
	if t.IsArray() {
		return Abi1RO(i) * x.regWidth(t.Elem())
	}
	if t.IsStruct() {
		k := Abi1RO(0)
		for j := 0; j < i; j++ {
			k += x.regWidth(t.FieldType(j))
		}
		return k
	}
	panic("Haven't implemented this case yet, do I need to?")
}

// at returns the register cursor for component i of t, where the first
// component is numbered 0.
func (c *registerCursor) at(t *types.Type, i int) registerCursor {
	rc := *c
	if i == 0 || c.regsLen == 0 {
		return rc
	}
	if t.IsArray() {
		w := c.config.NumParamRegs(t.Elem())
		rc.nextSlice += Abi1RO(i * w)
		return rc
	}
	if t.IsStruct() {
		for j := 0; j < i; j++ {
			rc.next(t.FieldType(j))
		}
		return rc
	}
	panic("Haven't implemented this case yet, do I need to?")
}

func (c *registerCursor) init(regs []abi.RegIndex, info *abi.ABIParamResultInfo, result *[]*Value, storeDest *Value) {
	c.regsLen = len(regs)
	c.nextSlice = 0
	if len(regs) == 0 {
		c.storeDest = storeDest // only save this if there are no registers, will explode if misused.
		return
	}
	c.config = info.Config()
	c.regValues = result
}

func (c *registerCursor) addArg(v *Value) {
	*c.regValues = append(*c.regValues, v)
}

func (c *registerCursor) hasRegs() bool {
	return c.regsLen > 0
}

type expandState struct {
	f                  *Func
	abi1               *abi.ABIConfig
	debug              int // odd values log lost statement markers, so likely settings are 1 (stmts), 2 (expansion), and 3 (both)
	canSSAType         func(*types.Type) bool
	regSize            int64
	sp                 *Value
	typs               *Types
	ptrSize            int64
	hiOffset           int64
	lowOffset          int64
	hiRo               Abi1RO
	loRo               Abi1RO
	namedSelects       map[*Value][]namedVal
	sdom               SparseTree
	commonSelectors    map[selKey]*Value // used to de-dupe selectors
	commonArgs         map[selKey]*Value // used to de-dupe OpArg/OpArgIntReg/OpArgFloatReg
	memForCall         map[ID]*Value     // For a call, need to know the unique selector that gets the mem.
	transformedSelects map[ID]bool       // OpSelectN after rewriting, either created or renumbered.
	indentLevel        int               // Indentation for debugging recursion
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
		(t.Size() > x.regSize && (t.IsInteger() || (x.f.Config.SoftFloat && t.IsFloat())))
}

// offsetFrom creates an offset from a pointer, simplifying chained offsets and offsets from SP
// TODO should also optimize offsets from SB?
func (x *expandState) offsetFrom(b *Block, from *Value, offset int64, pt *types.Type) *Value {
	ft := from.Type
	if offset == 0 {
		if ft == pt {
			return from
		}
		// This captures common, (apparently) safe cases.  The unsafe cases involve ft == uintptr
		if (ft.IsPtr() || ft.IsUnsafePtr()) && pt.IsPtr() {
			return from
		}
	}
	// Simplify, canonicalize
	for from.Op == OpOffPtr {
		offset += from.AuxInt
		from = from.Args[0]
	}
	if from == x.sp {
		return x.f.ConstOffPtrSP(pt, offset, x.sp)
	}
	return b.NewValue1I(from.Pos.WithNotStmt(), OpOffPtr, pt, offset, from)
}

// splitSlots splits one "field" (specified by sfx, offset, and ty) out of the LocalSlots in ls and returns the new LocalSlots this generates.
func (x *expandState) splitSlots(ls []*LocalSlot, sfx string, offset int64, ty *types.Type) []*LocalSlot {
	var locs []*LocalSlot
	for i := range ls {
		locs = append(locs, x.f.SplitSlot(ls[i], sfx, offset, ty))
	}
	return locs
}

// prAssignForArg returns the ABIParamAssignment for v, assumed to be an OpArg.
func (x *expandState) prAssignForArg(v *Value) *abi.ABIParamAssignment {
	if v.Op != OpArg {
		panic(badVal("Wanted OpArg, instead saw", v))
	}
	return ParamAssignmentForArgName(x.f, v.Aux.(*ir.Name))
}

// ParamAssignmentForArgName returns the ABIParamAssignment for f's arg with matching name.
func ParamAssignmentForArgName(f *Func, name *ir.Name) *abi.ABIParamAssignment {
	abiInfo := f.OwnAux.abiInfo
	ip := abiInfo.InParams()
	for i, a := range ip {
		if a.Name == name {
			return &ip[i]
		}
	}
	panic(fmt.Errorf("Did not match param %v in prInfo %+v", name, abiInfo.InParams()))
}

// indent increments (or decrements) the indentation.
func (x *expandState) indent(n int) {
	x.indentLevel += n
}

// Printf does an indented fmt.Printf on te format and args.
func (x *expandState) Printf(format string, a ...interface{}) (n int, err error) {
	if x.indentLevel > 0 {
		fmt.Printf("%[1]*s", x.indentLevel, "")
	}
	return fmt.Printf(format, a...)
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
func (x *expandState) rewriteSelect(leaf *Value, selector *Value, offset int64, regOffset Abi1RO) []*LocalSlot {
	if x.debug > 1 {
		x.indent(3)
		defer x.indent(-3)
		x.Printf("rewriteSelect(%s; %s; memOff=%d; regOff=%d)\n", leaf.LongString(), selector.LongString(), offset, regOffset)
	}
	var locs []*LocalSlot
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
	case OpArgIntReg, OpArgFloatReg:
		if leafType == selector.Type { // OpIData leads us here, sometimes.
			leaf.copyOf(selector)
		} else {
			x.f.Fatalf("Unexpected %s type, selector=%s, leaf=%s\n", selector.Op.String(), selector.LongString(), leaf.LongString())
		}
		if x.debug > 1 {
			x.Printf("---%s, break\n", selector.Op.String())
		}
	case OpArg:
		if !x.isAlreadyExpandedAggregateType(selector.Type) {
			if leafType == selector.Type { // OpIData leads us here, sometimes.
				x.newArgToMemOrRegs(selector, leaf, offset, regOffset, leafType, leaf.Pos)
			} else {
				x.f.Fatalf("Unexpected OpArg type, selector=%s, leaf=%s\n", selector.LongString(), leaf.LongString())
			}
			if x.debug > 1 {
				x.Printf("---OpArg, break\n")
			}
			break
		}
		switch leaf.Op {
		case OpIData, OpStructSelect, OpArraySelect:
			leafType = removeTrivialWrapperTypes(leaf.Type)
		}
		x.newArgToMemOrRegs(selector, leaf, offset, regOffset, leafType, leaf.Pos)

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
			if x.f.Config.SoftFloat && selector.Type.IsFloat() {
				if x.debug > 1 {
					x.Printf("---OpLoad, break\n")
				}
				break // softfloat pass will take care of that
			}
			x.f.Fatalf("Unexpected Load as selector, leaf=%s, selector=%s\n", leaf.LongString(), selector.LongString())
		}
		leaf.copyOf(selector)
		for _, s := range x.namedSelects[selector] {
			locs = append(locs, x.f.Names[s.locIndex])
		}

	case OpSelectN:
		// TODO(register args) result case
		// if applied to Op-mumble-call, the Aux tells us which result, regOffset specifies offset within result.  If a register, should rewrite to OpSelectN for new call.
		// TODO these may be duplicated. Should memoize. Intermediate selectors will go dead, no worries there.
		call := selector.Args[0]
		call0 := call
		aux := call.Aux.(*AuxCall)
		which := selector.AuxInt
		if x.transformedSelects[selector.ID] {
			// This is a minor hack.  Either this select has had its operand adjusted (mem) or
			// it is some other intermediate node that was rewritten to reference a register (not a generic arg).
			// This can occur with chains of selection/indexing from single field/element aggregates.
			leaf.copyOf(selector)
			break
		}
		if which == aux.NResults() { // mem is after the results.
			// rewrite v as a Copy of call -- the replacement call will produce a mem.
			if leaf != selector {
				panic(fmt.Errorf("Unexpected selector of memory, selector=%s, call=%s, leaf=%s", selector.LongString(), call.LongString(), leaf.LongString()))
			}
			if aux.abiInfo == nil {
				panic(badVal("aux.abiInfo nil for call", call))
			}
			if existing := x.memForCall[call.ID]; existing == nil {
				selector.AuxInt = int64(aux.abiInfo.OutRegistersUsed())
				x.memForCall[call.ID] = selector
				x.transformedSelects[selector.ID] = true // operand adjusted
			} else {
				selector.copyOf(existing)
			}

		} else {
			leafType := removeTrivialWrapperTypes(leaf.Type)
			if x.canSSAType(leafType) {
				pt := types.NewPtr(leafType)
				// Any selection right out of the arg area/registers has to be same Block as call, use call as mem input.
				// Create a "mem" for any loads that need to occur.
				if mem := x.memForCall[call.ID]; mem != nil {
					if mem.Block != call.Block {
						panic(fmt.Errorf("selector and call need to be in same block, selector=%s; call=%s", selector.LongString(), call.LongString()))
					}
					call = mem
				} else {
					mem = call.Block.NewValue1I(call.Pos.WithNotStmt(), OpSelectN, types.TypeMem, int64(aux.abiInfo.OutRegistersUsed()), call)
					x.transformedSelects[mem.ID] = true // select uses post-expansion indexing
					x.memForCall[call.ID] = mem
					call = mem
				}
				outParam := aux.abiInfo.OutParam(int(which))
				if len(outParam.Registers) > 0 {
					firstReg := uint32(0)
					for i := 0; i < int(which); i++ {
						firstReg += uint32(len(aux.abiInfo.OutParam(i).Registers))
					}
					reg := int64(regOffset + Abi1RO(firstReg))
					if leaf.Block == call.Block {
						leaf.reset(OpSelectN)
						leaf.SetArgs1(call0)
						leaf.Type = leafType
						leaf.AuxInt = reg
						x.transformedSelects[leaf.ID] = true // leaf, rewritten to use post-expansion indexing.
					} else {
						w := call.Block.NewValue1I(leaf.Pos, OpSelectN, leafType, reg, call0)
						x.transformedSelects[w.ID] = true // select, using post-expansion indexing.
						leaf.copyOf(w)
					}
				} else {
					off := x.offsetFrom(x.f.Entry, x.sp, offset+aux.OffsetOfResult(which), pt)
					if leaf.Block == call.Block {
						leaf.reset(OpLoad)
						leaf.SetArgs2(off, call)
						leaf.Type = leafType
					} else {
						w := call.Block.NewValue2(leaf.Pos, OpLoad, leafType, off, call)
						leaf.copyOf(w)
						if x.debug > 1 {
							x.Printf("---new %s\n", w.LongString())
						}
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
		var ls []*LocalSlot
		if w.Type.Kind() != types.TSTRUCT { // IData artifact
			ls = x.rewriteSelect(leaf, w, offset, regOffset)
		} else {
			fldi := int(selector.AuxInt)
			ls = x.rewriteSelect(leaf, w, offset+w.Type.FieldOff(fldi), regOffset+x.regOffset(w.Type, fldi))
			if w.Op != OpIData {
				for _, l := range ls {
					locs = append(locs, x.f.SplitStruct(l, int(selector.AuxInt)))
				}
			}
		}

	case OpArraySelect:
		w := selector.Args[0]
		index := selector.AuxInt
		x.rewriteSelect(leaf, w, offset+selector.Type.Size()*index, regOffset+x.regOffset(w.Type, int(index)))

	case OpInt64Hi:
		w := selector.Args[0]
		ls := x.rewriteSelect(leaf, w, offset+x.hiOffset, regOffset+x.hiRo)
		locs = x.splitSlots(ls, ".hi", x.hiOffset, leafType)

	case OpInt64Lo:
		w := selector.Args[0]
		ls := x.rewriteSelect(leaf, w, offset+x.lowOffset, regOffset+x.loRo)
		locs = x.splitSlots(ls, ".lo", x.lowOffset, leafType)

	case OpStringPtr:
		ls := x.rewriteSelect(leaf, selector.Args[0], offset, regOffset)
		locs = x.splitSlots(ls, ".ptr", 0, x.typs.BytePtr)

	case OpSlicePtr, OpSlicePtrUnchecked:
		w := selector.Args[0]
		ls := x.rewriteSelect(leaf, w, offset, regOffset)
		locs = x.splitSlots(ls, ".ptr", 0, types.NewPtr(w.Type.Elem()))

	case OpITab:
		w := selector.Args[0]
		ls := x.rewriteSelect(leaf, w, offset, regOffset)
		sfx := ".itab"
		if w.Type.IsEmptyInterface() {
			sfx = ".type"
		}
		locs = x.splitSlots(ls, sfx, 0, x.typs.Uintptr)

	case OpComplexReal:
		ls := x.rewriteSelect(leaf, selector.Args[0], offset, regOffset)
		locs = x.splitSlots(ls, ".real", 0, selector.Type)

	case OpComplexImag:
		ls := x.rewriteSelect(leaf, selector.Args[0], offset+selector.Type.Size(), regOffset+RO_complex_imag) // result is FloatNN, width of result is offset of imaginary part.
		locs = x.splitSlots(ls, ".imag", selector.Type.Size(), selector.Type)

	case OpStringLen, OpSliceLen:
		ls := x.rewriteSelect(leaf, selector.Args[0], offset+x.ptrSize, regOffset+RO_slice_len)
		locs = x.splitSlots(ls, ".len", x.ptrSize, leafType)

	case OpIData:
		ls := x.rewriteSelect(leaf, selector.Args[0], offset+x.ptrSize, regOffset+RO_iface_data)
		locs = x.splitSlots(ls, ".data", x.ptrSize, leafType)

	case OpSliceCap:
		ls := x.rewriteSelect(leaf, selector.Args[0], offset+2*x.ptrSize, regOffset+RO_slice_cap)
		locs = x.splitSlots(ls, ".cap", 2*x.ptrSize, leafType)

	case OpCopy: // If it's an intermediate result, recurse
		locs = x.rewriteSelect(leaf, selector.Args[0], offset, regOffset)
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
	dst := x.offsetFrom(b, base, offset, source.Type)
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

var indexNames [1]string = [1]string{"[0]"}

// pathTo returns the selection path to the leaf type at offset within container.
// e.g. len(thing.field[0]) => ".field[0].len"
// this is for purposes of generating names ultimately fed to a debugger.
func (x *expandState) pathTo(container, leaf *types.Type, offset int64) string {
	if container == leaf || offset == 0 && container.Size() == leaf.Size() {
		return ""
	}
	path := ""
outer:
	for {
		switch container.Kind() {
		case types.TARRAY:
			container = container.Elem()
			if container.Size() == 0 {
				return path
			}
			i := offset / container.Size()
			offset = offset % container.Size()
			// If a future compiler/ABI supports larger SSA/Arg-able arrays, expand indexNames.
			path = path + indexNames[i]
			continue
		case types.TSTRUCT:
			for i := 0; i < container.NumFields(); i++ {
				fld := container.Field(i)
				if fld.Offset+fld.Type.Size() > offset {
					offset -= fld.Offset
					path += "." + fld.Sym.Name
					container = fld.Type
					continue outer
				}
			}
			return path
		case types.TINT64, types.TUINT64:
			if container.Size() == x.regSize {
				return path
			}
			if offset == x.hiOffset {
				return path + ".hi"
			}
			return path + ".lo"
		case types.TINTER:
			if offset != 0 {
				return path + ".data"
			}
			if container.IsEmptyInterface() {
				return path + ".type"
			}
			return path + ".itab"

		case types.TSLICE:
			if offset == 2*x.regSize {
				return path + ".cap"
			}
			fallthrough
		case types.TSTRING:
			if offset == 0 {
				return path + ".ptr"
			}
			return path + ".len"
		case types.TCOMPLEX64, types.TCOMPLEX128:
			if offset == 0 {
				return path + ".real"
			}
			return path + ".imag"
		}
		return path
	}
}

// decomposeArg is a helper for storeArgOrLoad.
// It decomposes a Load or an Arg into smaller parts and returns the new mem.
// If the type does not match one of the expected aggregate types, it returns nil instead.
// Parameters:
//
//	pos           -- the location of any generated code.
//	b             -- the block into which any generated code should normally be placed
//	source        -- the value, possibly an aggregate, to be stored.
//	mem           -- the mem flowing into this decomposition (loads depend on it, stores updated it)
//	t             -- the type of the value to be stored
//	storeOffset   -- if the value is stored in memory, it is stored at base (see storeRc) + storeOffset
//	loadRegOffset -- regarding source as a value in registers, the register offset in ABI1.  Meaningful only if source is OpArg.
//	storeRc       -- storeRC; if the value is stored in registers, this specifies the registers.
//	                 StoreRc also identifies whether the target is registers or memory, and has the base for the store operation.
func (x *expandState) decomposeArg(pos src.XPos, b *Block, source, mem *Value, t *types.Type, storeOffset int64, loadRegOffset Abi1RO, storeRc registerCursor) *Value {

	pa := x.prAssignForArg(source)
	var locs []*LocalSlot
	for _, s := range x.namedSelects[source] {
		locs = append(locs, x.f.Names[s.locIndex])
	}

	if len(pa.Registers) > 0 {
		// Handle the in-registers case directly
		rts, offs := pa.RegisterTypesAndOffsets()
		last := loadRegOffset + x.regWidth(t)
		if offs[loadRegOffset] != 0 {
			// Document the problem before panicking.
			for i := 0; i < len(rts); i++ {
				rt := rts[i]
				off := offs[i]
				fmt.Printf("rt=%s, off=%d, rt.Width=%d, rt.Align=%d\n", rt.String(), off, rt.Size(), uint8(rt.Alignment()))
			}
			panic(fmt.Errorf("offset %d of requested register %d should be zero, source=%s", offs[loadRegOffset], loadRegOffset, source.LongString()))
		}

		if x.debug > 1 {
			x.Printf("decompose arg %s has %d locs\n", source.LongString(), len(locs))
		}

		for i := loadRegOffset; i < last; i++ {
			rt := rts[i]
			off := offs[i]
			w := x.commonArgs[selKey{source, off, rt.Size(), rt}]
			if w == nil {
				w = x.newArgToMemOrRegs(source, w, off, i, rt, pos)
				suffix := x.pathTo(source.Type, rt, off)
				if suffix != "" {
					x.splitSlotsIntoNames(locs, suffix, off, rt, w)
				}
			}
			if t.IsPtrShaped() {
				// Preserve the original store type. This ensures pointer type
				// properties aren't discarded (e.g, notinheap).
				if rt.Size() != t.Size() || len(pa.Registers) != 1 || i != loadRegOffset {
					b.Func.Fatalf("incompatible store type %v and %v, i=%d", t, rt, i)
				}
				rt = t
			}
			mem = x.storeArgOrLoad(pos, b, w, mem, rt, storeOffset+off, i, storeRc.next(rt))
		}
		return mem
	}

	u := source.Type
	switch u.Kind() {
	case types.TARRAY:
		elem := u.Elem()
		elemRO := x.regWidth(elem)
		for i := int64(0); i < u.NumElem(); i++ {
			elemOff := i * elem.Size()
			mem = storeOneArg(x, pos, b, locs, indexNames[i], source, mem, elem, elemOff, storeOffset+elemOff, loadRegOffset, storeRc.next(elem))
			loadRegOffset += elemRO
			pos = pos.WithNotStmt()
		}
		return mem
	case types.TSTRUCT:
		for i := 0; i < u.NumFields(); i++ {
			fld := u.Field(i)
			mem = storeOneArg(x, pos, b, locs, "."+fld.Sym.Name, source, mem, fld.Type, fld.Offset, storeOffset+fld.Offset, loadRegOffset, storeRc.next(fld.Type))
			loadRegOffset += x.regWidth(fld.Type)
			pos = pos.WithNotStmt()
		}
		return mem
	case types.TINT64, types.TUINT64:
		if t.Size() == x.regSize {
			break
		}
		tHi, tLo := x.intPairTypes(t.Kind())
		mem = storeOneArg(x, pos, b, locs, ".hi", source, mem, tHi, x.hiOffset, storeOffset+x.hiOffset, loadRegOffset+x.hiRo, storeRc.plus(x.hiRo))
		pos = pos.WithNotStmt()
		return storeOneArg(x, pos, b, locs, ".lo", source, mem, tLo, x.lowOffset, storeOffset+x.lowOffset, loadRegOffset+x.loRo, storeRc.plus(x.loRo))
	case types.TINTER:
		sfx := ".itab"
		if u.IsEmptyInterface() {
			sfx = ".type"
		}
		return storeTwoArg(x, pos, b, locs, sfx, ".idata", source, mem, x.typs.Uintptr, x.typs.BytePtr, 0, storeOffset, loadRegOffset, storeRc)
	case types.TSTRING:
		return storeTwoArg(x, pos, b, locs, ".ptr", ".len", source, mem, x.typs.BytePtr, x.typs.Int, 0, storeOffset, loadRegOffset, storeRc)
	case types.TCOMPLEX64:
		return storeTwoArg(x, pos, b, locs, ".real", ".imag", source, mem, x.typs.Float32, x.typs.Float32, 0, storeOffset, loadRegOffset, storeRc)
	case types.TCOMPLEX128:
		return storeTwoArg(x, pos, b, locs, ".real", ".imag", source, mem, x.typs.Float64, x.typs.Float64, 0, storeOffset, loadRegOffset, storeRc)
	case types.TSLICE:
		mem = storeOneArg(x, pos, b, locs, ".ptr", source, mem, x.typs.BytePtr, 0, storeOffset, loadRegOffset, storeRc.next(x.typs.BytePtr))
		return storeTwoArg(x, pos, b, locs, ".len", ".cap", source, mem, x.typs.Int, x.typs.Int, x.ptrSize, storeOffset+x.ptrSize, loadRegOffset+RO_slice_len, storeRc)
	}
	return nil
}

func (x *expandState) splitSlotsIntoNames(locs []*LocalSlot, suffix string, off int64, rt *types.Type, w *Value) {
	wlocs := x.splitSlots(locs, suffix, off, rt)
	for _, l := range wlocs {
		old, ok := x.f.NamedValues[*l]
		x.f.NamedValues[*l] = append(old, w)
		if !ok {
			x.f.Names = append(x.f.Names, l)
		}
	}
}

// decomposeLoad is a helper for storeArgOrLoad.
// It decomposes a Load  into smaller parts and returns the new mem.
// If the type does not match one of the expected aggregate types, it returns nil instead.
// Parameters:
//
//	pos           -- the location of any generated code.
//	b             -- the block into which any generated code should normally be placed
//	source        -- the value, possibly an aggregate, to be stored.
//	mem           -- the mem flowing into this decomposition (loads depend on it, stores updated it)
//	t             -- the type of the value to be stored
//	storeOffset   -- if the value is stored in memory, it is stored at base (see storeRc) + offset
//	loadRegOffset -- regarding source as a value in registers, the register offset in ABI1.  Meaningful only if source is OpArg.
//	storeRc       -- storeRC; if the value is stored in registers, this specifies the registers.
//	                 StoreRc also identifies whether the target is registers or memory, and has the base for the store operation.
//
// TODO -- this needs cleanup; it just works for SSA-able aggregates, and won't fully generalize to register-args aggregates.
func (x *expandState) decomposeLoad(pos src.XPos, b *Block, source, mem *Value, t *types.Type, storeOffset int64, loadRegOffset Abi1RO, storeRc registerCursor) *Value {
	u := source.Type
	switch u.Kind() {
	case types.TARRAY:
		elem := u.Elem()
		elemRO := x.regWidth(elem)
		for i := int64(0); i < u.NumElem(); i++ {
			elemOff := i * elem.Size()
			mem = storeOneLoad(x, pos, b, source, mem, elem, elemOff, storeOffset+elemOff, loadRegOffset, storeRc.next(elem))
			loadRegOffset += elemRO
			pos = pos.WithNotStmt()
		}
		return mem
	case types.TSTRUCT:
		for i := 0; i < u.NumFields(); i++ {
			fld := u.Field(i)
			mem = storeOneLoad(x, pos, b, source, mem, fld.Type, fld.Offset, storeOffset+fld.Offset, loadRegOffset, storeRc.next(fld.Type))
			loadRegOffset += x.regWidth(fld.Type)
			pos = pos.WithNotStmt()
		}
		return mem
	case types.TINT64, types.TUINT64:
		if t.Size() == x.regSize {
			break
		}
		tHi, tLo := x.intPairTypes(t.Kind())
		mem = storeOneLoad(x, pos, b, source, mem, tHi, x.hiOffset, storeOffset+x.hiOffset, loadRegOffset+x.hiRo, storeRc.plus(x.hiRo))
		pos = pos.WithNotStmt()
		return storeOneLoad(x, pos, b, source, mem, tLo, x.lowOffset, storeOffset+x.lowOffset, loadRegOffset+x.loRo, storeRc.plus(x.loRo))
	case types.TINTER:
		return storeTwoLoad(x, pos, b, source, mem, x.typs.Uintptr, x.typs.BytePtr, 0, storeOffset, loadRegOffset, storeRc)
	case types.TSTRING:
		return storeTwoLoad(x, pos, b, source, mem, x.typs.BytePtr, x.typs.Int, 0, storeOffset, loadRegOffset, storeRc)
	case types.TCOMPLEX64:
		return storeTwoLoad(x, pos, b, source, mem, x.typs.Float32, x.typs.Float32, 0, storeOffset, loadRegOffset, storeRc)
	case types.TCOMPLEX128:
		return storeTwoLoad(x, pos, b, source, mem, x.typs.Float64, x.typs.Float64, 0, storeOffset, loadRegOffset, storeRc)
	case types.TSLICE:
		mem = storeOneLoad(x, pos, b, source, mem, x.typs.BytePtr, 0, storeOffset, loadRegOffset, storeRc.next(x.typs.BytePtr))
		return storeTwoLoad(x, pos, b, source, mem, x.typs.Int, x.typs.Int, x.ptrSize, storeOffset+x.ptrSize, loadRegOffset+RO_slice_len, storeRc)
	}
	return nil
}

// storeOneArg creates a decomposed (one step) arg that is then stored.
// pos and b locate the store instruction, source is the "base" of the value input,
// mem is the input mem, t is the type in question, and offArg and offStore are the offsets from the respective bases.
func storeOneArg(x *expandState, pos src.XPos, b *Block, locs []*LocalSlot, suffix string, source, mem *Value, t *types.Type, argOffset, storeOffset int64, loadRegOffset Abi1RO, storeRc registerCursor) *Value {
	if x.debug > 1 {
		x.indent(3)
		defer x.indent(-3)
		x.Printf("storeOneArg(%s;  %s;  %s; aO=%d; sO=%d; lrO=%d; %s)\n", source.LongString(), mem.String(), t.String(), argOffset, storeOffset, loadRegOffset, storeRc.String())
	}

	w := x.commonArgs[selKey{source, argOffset, t.Size(), t}]
	if w == nil {
		w = x.newArgToMemOrRegs(source, w, argOffset, loadRegOffset, t, pos)
		x.splitSlotsIntoNames(locs, suffix, argOffset, t, w)
	}
	return x.storeArgOrLoad(pos, b, w, mem, t, storeOffset, loadRegOffset, storeRc)
}

// storeOneLoad creates a decomposed (one step) load that is then stored.
func storeOneLoad(x *expandState, pos src.XPos, b *Block, source, mem *Value, t *types.Type, offArg, offStore int64, loadRegOffset Abi1RO, storeRc registerCursor) *Value {
	from := x.offsetFrom(source.Block, source.Args[0], offArg, types.NewPtr(t))
	w := source.Block.NewValue2(source.Pos, OpLoad, t, from, mem)
	return x.storeArgOrLoad(pos, b, w, mem, t, offStore, loadRegOffset, storeRc)
}

func storeTwoArg(x *expandState, pos src.XPos, b *Block, locs []*LocalSlot, suffix1 string, suffix2 string, source, mem *Value, t1, t2 *types.Type, offArg, offStore int64, loadRegOffset Abi1RO, storeRc registerCursor) *Value {
	mem = storeOneArg(x, pos, b, locs, suffix1, source, mem, t1, offArg, offStore, loadRegOffset, storeRc.next(t1))
	pos = pos.WithNotStmt()
	t1Size := t1.Size()
	return storeOneArg(x, pos, b, locs, suffix2, source, mem, t2, offArg+t1Size, offStore+t1Size, loadRegOffset+1, storeRc)
}

// storeTwoLoad creates a pair of decomposed (one step) loads that are then stored.
// the elements of the pair must not require any additional alignment.
func storeTwoLoad(x *expandState, pos src.XPos, b *Block, source, mem *Value, t1, t2 *types.Type, offArg, offStore int64, loadRegOffset Abi1RO, storeRc registerCursor) *Value {
	mem = storeOneLoad(x, pos, b, source, mem, t1, offArg, offStore, loadRegOffset, storeRc.next(t1))
	pos = pos.WithNotStmt()
	t1Size := t1.Size()
	return storeOneLoad(x, pos, b, source, mem, t2, offArg+t1Size, offStore+t1Size, loadRegOffset+1, storeRc)
}

// storeArgOrLoad converts stores of SSA-able potentially aggregatable arguments (passed to a call) into a series of primitive-typed
// stores of non-aggregate types.  It recursively walks up a chain of selectors until it reaches a Load or an Arg.
// If it does not reach a Load or an Arg, nothing happens; this allows a little freedom in phase ordering.
func (x *expandState) storeArgOrLoad(pos src.XPos, b *Block, source, mem *Value, t *types.Type, storeOffset int64, loadRegOffset Abi1RO, storeRc registerCursor) *Value {
	if x.debug > 1 {
		x.indent(3)
		defer x.indent(-3)
		x.Printf("storeArgOrLoad(%s;  %s;  %s; %d; %s)\n", source.LongString(), mem.String(), t.String(), storeOffset, storeRc.String())
	}

	// Start with Opcodes that can be disassembled
	switch source.Op {
	case OpCopy:
		return x.storeArgOrLoad(pos, b, source.Args[0], mem, t, storeOffset, loadRegOffset, storeRc)

	case OpLoad, OpDereference:
		ret := x.decomposeLoad(pos, b, source, mem, t, storeOffset, loadRegOffset, storeRc)
		if ret != nil {
			return ret
		}

	case OpArg:
		ret := x.decomposeArg(pos, b, source, mem, t, storeOffset, loadRegOffset, storeRc)
		if ret != nil {
			return ret
		}

	case OpArrayMake0, OpStructMake0:
		// TODO(register args) is this correct for registers?
		return mem

	case OpStructMake1, OpStructMake2, OpStructMake3, OpStructMake4:
		for i := 0; i < t.NumFields(); i++ {
			fld := t.Field(i)
			mem = x.storeArgOrLoad(pos, b, source.Args[i], mem, fld.Type, storeOffset+fld.Offset, 0, storeRc.next(fld.Type))
			pos = pos.WithNotStmt()
		}
		return mem

	case OpArrayMake1:
		return x.storeArgOrLoad(pos, b, source.Args[0], mem, t.Elem(), storeOffset, 0, storeRc.at(t, 0))

	case OpInt64Make:
		tHi, tLo := x.intPairTypes(t.Kind())
		mem = x.storeArgOrLoad(pos, b, source.Args[0], mem, tHi, storeOffset+x.hiOffset, 0, storeRc.next(tHi))
		pos = pos.WithNotStmt()
		return x.storeArgOrLoad(pos, b, source.Args[1], mem, tLo, storeOffset+x.lowOffset, 0, storeRc)

	case OpComplexMake:
		tPart := x.typs.Float32
		wPart := t.Size() / 2
		if wPart == 8 {
			tPart = x.typs.Float64
		}
		mem = x.storeArgOrLoad(pos, b, source.Args[0], mem, tPart, storeOffset, 0, storeRc.next(tPart))
		pos = pos.WithNotStmt()
		return x.storeArgOrLoad(pos, b, source.Args[1], mem, tPart, storeOffset+wPart, 0, storeRc)

	case OpIMake:
		mem = x.storeArgOrLoad(pos, b, source.Args[0], mem, x.typs.Uintptr, storeOffset, 0, storeRc.next(x.typs.Uintptr))
		pos = pos.WithNotStmt()
		return x.storeArgOrLoad(pos, b, source.Args[1], mem, x.typs.BytePtr, storeOffset+x.ptrSize, 0, storeRc)

	case OpStringMake:
		mem = x.storeArgOrLoad(pos, b, source.Args[0], mem, x.typs.BytePtr, storeOffset, 0, storeRc.next(x.typs.BytePtr))
		pos = pos.WithNotStmt()
		return x.storeArgOrLoad(pos, b, source.Args[1], mem, x.typs.Int, storeOffset+x.ptrSize, 0, storeRc)

	case OpSliceMake:
		mem = x.storeArgOrLoad(pos, b, source.Args[0], mem, x.typs.BytePtr, storeOffset, 0, storeRc.next(x.typs.BytePtr))
		pos = pos.WithNotStmt()
		mem = x.storeArgOrLoad(pos, b, source.Args[1], mem, x.typs.Int, storeOffset+x.ptrSize, 0, storeRc.next(x.typs.Int))
		return x.storeArgOrLoad(pos, b, source.Args[2], mem, x.typs.Int, storeOffset+2*x.ptrSize, 0, storeRc)
	}

	// For nodes that cannot be taken apart -- OpSelectN, other structure selectors.
	switch t.Kind() {
	case types.TARRAY:
		elt := t.Elem()
		if source.Type != t && t.NumElem() == 1 && elt.Size() == t.Size() && t.Size() == x.regSize {
			t = removeTrivialWrapperTypes(t)
			// it could be a leaf type, but the "leaf" could be complex64 (for example)
			return x.storeArgOrLoad(pos, b, source, mem, t, storeOffset, loadRegOffset, storeRc)
		}
		eltRO := x.regWidth(elt)
		source.Type = t
		for i := int64(0); i < t.NumElem(); i++ {
			sel := source.Block.NewValue1I(pos, OpArraySelect, elt, i, source)
			mem = x.storeArgOrLoad(pos, b, sel, mem, elt, storeOffset+i*elt.Size(), loadRegOffset, storeRc.at(t, 0))
			loadRegOffset += eltRO
			pos = pos.WithNotStmt()
		}
		return mem

	case types.TSTRUCT:
		if source.Type != t && t.NumFields() == 1 && t.Field(0).Type.Size() == t.Size() && t.Size() == x.regSize {
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
			return x.storeArgOrLoad(pos, b, source, mem, t, storeOffset, loadRegOffset, storeRc)
		}

		source.Type = t
		for i := 0; i < t.NumFields(); i++ {
			fld := t.Field(i)
			sel := source.Block.NewValue1I(pos, OpStructSelect, fld.Type, int64(i), source)
			mem = x.storeArgOrLoad(pos, b, sel, mem, fld.Type, storeOffset+fld.Offset, loadRegOffset, storeRc.next(fld.Type))
			loadRegOffset += x.regWidth(fld.Type)
			pos = pos.WithNotStmt()
		}
		return mem

	case types.TINT64, types.TUINT64:
		if t.Size() == x.regSize {
			break
		}
		tHi, tLo := x.intPairTypes(t.Kind())
		sel := source.Block.NewValue1(pos, OpInt64Hi, tHi, source)
		mem = x.storeArgOrLoad(pos, b, sel, mem, tHi, storeOffset+x.hiOffset, loadRegOffset+x.hiRo, storeRc.plus(x.hiRo))
		pos = pos.WithNotStmt()
		sel = source.Block.NewValue1(pos, OpInt64Lo, tLo, source)
		return x.storeArgOrLoad(pos, b, sel, mem, tLo, storeOffset+x.lowOffset, loadRegOffset+x.loRo, storeRc.plus(x.hiRo))

	case types.TINTER:
		sel := source.Block.NewValue1(pos, OpITab, x.typs.BytePtr, source)
		mem = x.storeArgOrLoad(pos, b, sel, mem, x.typs.BytePtr, storeOffset, loadRegOffset, storeRc.next(x.typs.BytePtr))
		pos = pos.WithNotStmt()
		sel = source.Block.NewValue1(pos, OpIData, x.typs.BytePtr, source)
		return x.storeArgOrLoad(pos, b, sel, mem, x.typs.BytePtr, storeOffset+x.ptrSize, loadRegOffset+RO_iface_data, storeRc)

	case types.TSTRING:
		sel := source.Block.NewValue1(pos, OpStringPtr, x.typs.BytePtr, source)
		mem = x.storeArgOrLoad(pos, b, sel, mem, x.typs.BytePtr, storeOffset, loadRegOffset, storeRc.next(x.typs.BytePtr))
		pos = pos.WithNotStmt()
		sel = source.Block.NewValue1(pos, OpStringLen, x.typs.Int, source)
		return x.storeArgOrLoad(pos, b, sel, mem, x.typs.Int, storeOffset+x.ptrSize, loadRegOffset+RO_string_len, storeRc)

	case types.TSLICE:
		et := types.NewPtr(t.Elem())
		sel := source.Block.NewValue1(pos, OpSlicePtr, et, source)
		mem = x.storeArgOrLoad(pos, b, sel, mem, et, storeOffset, loadRegOffset, storeRc.next(et))
		pos = pos.WithNotStmt()
		sel = source.Block.NewValue1(pos, OpSliceLen, x.typs.Int, source)
		mem = x.storeArgOrLoad(pos, b, sel, mem, x.typs.Int, storeOffset+x.ptrSize, loadRegOffset+RO_slice_len, storeRc.next(x.typs.Int))
		sel = source.Block.NewValue1(pos, OpSliceCap, x.typs.Int, source)
		return x.storeArgOrLoad(pos, b, sel, mem, x.typs.Int, storeOffset+2*x.ptrSize, loadRegOffset+RO_slice_cap, storeRc)

	case types.TCOMPLEX64:
		sel := source.Block.NewValue1(pos, OpComplexReal, x.typs.Float32, source)
		mem = x.storeArgOrLoad(pos, b, sel, mem, x.typs.Float32, storeOffset, loadRegOffset, storeRc.next(x.typs.Float32))
		pos = pos.WithNotStmt()
		sel = source.Block.NewValue1(pos, OpComplexImag, x.typs.Float32, source)
		return x.storeArgOrLoad(pos, b, sel, mem, x.typs.Float32, storeOffset+4, loadRegOffset+RO_complex_imag, storeRc)

	case types.TCOMPLEX128:
		sel := source.Block.NewValue1(pos, OpComplexReal, x.typs.Float64, source)
		mem = x.storeArgOrLoad(pos, b, sel, mem, x.typs.Float64, storeOffset, loadRegOffset, storeRc.next(x.typs.Float64))
		pos = pos.WithNotStmt()
		sel = source.Block.NewValue1(pos, OpComplexImag, x.typs.Float64, source)
		return x.storeArgOrLoad(pos, b, sel, mem, x.typs.Float64, storeOffset+8, loadRegOffset+RO_complex_imag, storeRc)
	}

	s := mem
	if source.Op == OpDereference {
		source.Op = OpLoad // For purposes of parameter passing expansion, a Dereference is a Load.
	}
	if storeRc.hasRegs() {
		storeRc.addArg(source)
	} else {
		dst := x.offsetFrom(b, storeRc.storeDest, storeOffset, types.NewPtr(t))
		s = b.NewValue3A(pos, OpStore, types.TypeMem, t, dst, source, mem)
	}
	if x.debug > 1 {
		x.Printf("-->storeArg returns %s, storeRc=%s\n", s.LongString(), storeRc.String())
	}
	return s
}

// rewriteArgs replaces all the call-parameter Args to a call with their register translation (if any).
// Preceding parameters (code pointers, closure pointer) are preserved, and the memory input is modified
// to account for any parameter stores required.
// Any of the old Args that have their use count fall to zero are marked OpInvalid.
func (x *expandState) rewriteArgs(v *Value, firstArg int) {
	if x.debug > 1 {
		x.indent(3)
		defer x.indent(-3)
		x.Printf("rewriteArgs(%s; %d)\n", v.LongString(), firstArg)
	}
	// Thread the stores on the memory arg
	aux := v.Aux.(*AuxCall)
	m0 := v.MemoryArg()
	mem := m0
	newArgs := []*Value{}
	oldArgs := []*Value{}
	sp := x.sp
	if v.Op == OpTailLECall {
		// For tail call, we unwind the frame before the call so we'll use the caller's
		// SP.
		sp = x.f.Entry.NewValue0(src.NoXPos, OpGetCallerSP, x.typs.Uintptr)
	}
	for i, a := range v.Args[firstArg : len(v.Args)-1] { // skip leading non-parameter SSA Args and trailing mem SSA Arg.
		oldArgs = append(oldArgs, a)
		auxI := int64(i)
		aRegs := aux.RegsOfArg(auxI)
		aType := aux.TypeOfArg(auxI)
		if len(aRegs) == 0 && a.Op == OpDereference {
			aOffset := aux.OffsetOfArg(auxI)
			if a.MemoryArg() != m0 {
				x.f.Fatalf("Op...LECall and OpDereference have mismatched mem, %s and %s", v.LongString(), a.LongString())
			}
			if v.Op == OpTailLECall {
				// It's common for a tail call passing the same arguments (e.g. method wrapper),
				// so this would be a self copy. Detect this and optimize it out.
				a0 := a.Args[0]
				if a0.Op == OpLocalAddr {
					n := a0.Aux.(*ir.Name)
					if n.Class == ir.PPARAM && n.FrameOffset()+x.f.Config.ctxt.Arch.FixedFrameSize == aOffset {
						continue
					}
				}
			}
			// "Dereference" of addressed (probably not-SSA-eligible) value becomes Move
			// TODO(register args) this will be more complicated with registers in the picture.
			mem = x.rewriteDereference(v.Block, sp, a, mem, aOffset, aux.SizeOfArg(auxI), aType, a.Pos)
		} else {
			var rc registerCursor
			var result *[]*Value
			var aOffset int64
			if len(aRegs) > 0 {
				result = &newArgs
			} else {
				aOffset = aux.OffsetOfArg(auxI)
			}
			if v.Op == OpTailLECall && a.Op == OpArg && a.AuxInt == 0 {
				// It's common for a tail call passing the same arguments (e.g. method wrapper),
				// so this would be a self copy. Detect this and optimize it out.
				n := a.Aux.(*ir.Name)
				if n.Class == ir.PPARAM && n.FrameOffset()+x.f.Config.ctxt.Arch.FixedFrameSize == aOffset {
					continue
				}
			}
			if x.debug > 1 {
				x.Printf("...storeArg %s, %v, %d\n", a.LongString(), aType, aOffset)
			}
			rc.init(aRegs, aux.abiInfo, result, sp)
			mem = x.storeArgOrLoad(a.Pos, v.Block, a, mem, aType, aOffset, 0, rc)
		}
	}
	var preArgStore [2]*Value
	preArgs := append(preArgStore[:0], v.Args[0:firstArg]...)
	v.resetArgs()
	v.AddArgs(preArgs...)
	v.AddArgs(newArgs...)
	v.AddArg(mem)
	for _, a := range oldArgs {
		if a.Uses == 0 {
			x.invalidateRecursively(a)
		}
	}

	return
}

func (x *expandState) invalidateRecursively(a *Value) {
	var s string
	if x.debug > 0 {
		plus := " "
		if a.Pos.IsStmt() == src.PosIsStmt {
			plus = " +"
		}
		s = a.String() + plus + a.Pos.LineNumber() + " " + a.LongString()
		if x.debug > 1 {
			x.Printf("...marking %v unused\n", s)
		}
	}
	lost := a.invalidateRecursively()
	if x.debug&1 != 0 && lost { // For odd values of x.debug, do this.
		x.Printf("Lost statement marker in %s on former %s\n", base.Ctxt.Pkgpath+"."+x.f.Name, s)
	}
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
		f:                  f,
		abi1:               f.ABI1,
		debug:              f.pass.debug,
		canSSAType:         f.fe.CanSSA,
		regSize:            f.Config.RegSize,
		sp:                 sp,
		typs:               &f.Config.Types,
		ptrSize:            f.Config.PtrSize,
		namedSelects:       make(map[*Value][]namedVal),
		sdom:               f.Sdom(),
		commonArgs:         make(map[selKey]*Value),
		memForCall:         make(map[ID]*Value),
		transformedSelects: make(map[ID]bool),
	}

	// For 32-bit, need to deal with decomposition of 64-bit integers, which depends on endianness.
	if f.Config.BigEndian {
		x.lowOffset, x.hiOffset = 4, 0
		x.loRo, x.hiRo = 1, 0
	} else {
		x.lowOffset, x.hiOffset = 0, 4
		x.loRo, x.hiRo = 0, 1
	}

	if x.debug > 1 {
		x.Printf("\nexpandsCalls(%s)\n", f.Name)
	}

	for i, name := range f.Names {
		t := name.Type
		if x.isAlreadyExpandedAggregateType(t) {
			for j, v := range f.NamedValues[*name] {
				if v.Op == OpSelectN || v.Op == OpArg && x.isAlreadyExpandedAggregateType(v.Type) {
					ns := x.namedSelects[v]
					x.namedSelects[v] = append(ns, namedVal{locIndex: i, valIndex: j})
				}
			}
		}
	}

	// TODO if too slow, whole program iteration can be replaced w/ slices of appropriate values, accumulated in first loop here.

	// Step 0: rewrite the calls to convert args to calls into stores/register movement.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			firstArg := 0
			switch v.Op {
			case OpStaticLECall, OpTailLECall:
			case OpInterLECall:
				firstArg = 1
			case OpClosureLECall:
				firstArg = 2
			default:
				continue
			}
			x.rewriteArgs(v, firstArg)
		}
		if isBlockMultiValueExit(b) {
			x.indent(3)
			// Very similar to code in rewriteArgs, but results instead of args.
			v := b.Controls[0]
			m0 := v.MemoryArg()
			mem := m0
			aux := f.OwnAux
			allResults := []*Value{}
			if x.debug > 1 {
				x.Printf("multiValueExit rewriting %s\n", v.LongString())
			}
			var oldArgs []*Value
			for j, a := range v.Args[:len(v.Args)-1] {
				oldArgs = append(oldArgs, a)
				i := int64(j)
				auxType := aux.TypeOfResult(i)
				auxBase := b.NewValue2A(v.Pos, OpLocalAddr, types.NewPtr(auxType), aux.NameOfResult(i), x.sp, mem)
				auxOffset := int64(0)
				auxSize := aux.SizeOfResult(i)
				aRegs := aux.RegsOfResult(int64(j))
				if len(aRegs) == 0 && a.Op == OpDereference {
					// Avoid a self-move, and if one is detected try to remove the already-inserted VarDef for the assignment that won't happen.
					if dAddr, dMem := a.Args[0], a.Args[1]; dAddr.Op == OpLocalAddr && dAddr.Args[0].Op == OpSP &&
						dAddr.Args[1] == dMem && dAddr.Aux == aux.NameOfResult(i) {
						if dMem.Op == OpVarDef && dMem.Aux == dAddr.Aux {
							dMem.copyOf(dMem.MemoryArg()) // elide the VarDef
						}
						continue
					}
					mem = x.rewriteDereference(v.Block, auxBase, a, mem, auxOffset, auxSize, auxType, a.Pos)
				} else {
					if a.Op == OpLoad && a.Args[0].Op == OpLocalAddr {
						addr := a.Args[0] // This is a self-move. // TODO(register args) do what here for registers?
						if addr.MemoryArg() == a.MemoryArg() && addr.Aux == aux.NameOfResult(i) {
							continue
						}
					}
					var rc registerCursor
					var result *[]*Value
					if len(aRegs) > 0 {
						result = &allResults
					}
					rc.init(aRegs, aux.abiInfo, result, auxBase)
					mem = x.storeArgOrLoad(v.Pos, b, a, mem, aux.TypeOfResult(i), auxOffset, 0, rc)
				}
			}
			v.resetArgs()
			v.AddArgs(allResults...)
			v.AddArg(mem)
			v.Type = types.NewResults(append(abi.RegisterTypes(aux.abiInfo.OutParams()), types.TypeMem))
			b.SetControl(v)
			for _, a := range oldArgs {
				if a.Uses == 0 {
					if x.debug > 1 {
						x.Printf("...marking %v unused\n", a.LongString())
					}
					x.invalidateRecursively(a)
				}
			}
			if x.debug > 1 {
				x.Printf("...multiValueExit new result %s\n", v.LongString())
			}
			x.indent(-3)
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
				dst, mem := v.Args[0], v.Args[2]
				mem = x.storeArgOrLoad(v.Pos, b, source, mem, t, 0, 0, registerCursor{storeDest: dst})
				v.copyOf(mem)
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
				OpSlicePtr, OpSliceLen, OpSliceCap, OpSlicePtrUnchecked,
				OpComplexReal, OpComplexImag,
				OpInt64Hi, OpInt64Lo:
				w := v.Args[0]
				switch w.Op {
				case OpStructSelect, OpArraySelect, OpSelectN, OpArg:
					val2Preds[w] += 1
					if x.debug > 1 {
						x.Printf("v2p[%s] = %d\n", w.LongString(), val2Preds[w])
					}
				}
				fallthrough

			case OpSelectN:
				if _, ok := val2Preds[v]; !ok {
					val2Preds[v] = 0
					if x.debug > 1 {
						x.Printf("v2p[%s] = %d\n", v.LongString(), val2Preds[v])
					}
				}

			case OpArg:
				if !x.isAlreadyExpandedAggregateType(v.Type) {
					continue
				}
				if _, ok := val2Preds[v]; !ok {
					val2Preds[v] = 0
					if x.debug > 1 {
						x.Printf("v2p[%s] = %d\n", v.LongString(), val2Preds[v])
					}
				}

			case OpSelectNAddr:
				// Do these directly, there are no chains of selectors.
				call := v.Args[0]
				which := v.AuxInt
				aux := call.Aux.(*AuxCall)
				pt := v.Type
				off := x.offsetFrom(x.f.Entry, x.sp, aux.OffsetOfResult(which), pt)
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

	x.commonSelectors = make(map[selKey]*Value)
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
		size := typ.Size()
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
			offset = v.AuxInt // offset is just a key, really.
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
		sk := selKey{from: w, size: size, offsetOrIndex: offset, typ: typ}
		dupe := x.commonSelectors[sk]
		if dupe == nil {
			x.commonSelectors[sk] = v
		} else if x.sdom.IsAncestorEq(dupe.Block, v.Block) {
			if x.debug > 1 {
				x.Printf("Duplicate, make %s copy of %s\n", v, dupe)
			}
			v.copyOf(dupe)
		} else {
			// Because values are processed in dominator order, the old common[s] will never dominate after a miss is seen.
			// Installing the new value might match some future values.
			x.commonSelectors[sk] = v
		}
	}

	// Indices of entries in f.Names that need to be deleted.
	var toDelete []namedVal

	// Rewrite selectors.
	for i, v := range allOrdered {
		if x.debug > 1 {
			b := v.Block
			x.Printf("allOrdered[%d] = b%d, %s, uses=%d\n", i, b.ID, v.LongString(), v.Uses)
		}
		if v.Uses == 0 {
			x.invalidateRecursively(v)
			continue
		}
		if v.Op == OpCopy {
			continue
		}
		locs := x.rewriteSelect(v, v, 0, 0)
		// Install new names.
		if v.Type.IsMemory() {
			continue
		}
		// Leaf types may have debug locations
		if !x.isAlreadyExpandedAggregateType(v.Type) {
			for _, l := range locs {
				if _, ok := f.NamedValues[*l]; !ok {
					f.Names = append(f.Names, l)
				}
				f.NamedValues[*l] = append(f.NamedValues[*l], v)
			}
			continue
		}
		if ns, ok := x.namedSelects[v]; ok {
			// Not-leaf types that had debug locations need to lose them.

			toDelete = append(toDelete, ns...)
		}
	}

	deleteNamedVals(f, toDelete)

	// Step 4: rewrite the calls themselves, correcting the type.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			switch v.Op {
			case OpArg:
				x.rewriteArgToMemOrRegs(v)
			case OpStaticLECall:
				v.Op = OpStaticCall
				rts := abi.RegisterTypes(v.Aux.(*AuxCall).abiInfo.OutParams())
				v.Type = types.NewResults(append(rts, types.TypeMem))
			case OpTailLECall:
				v.Op = OpTailCall
				rts := abi.RegisterTypes(v.Aux.(*AuxCall).abiInfo.OutParams())
				v.Type = types.NewResults(append(rts, types.TypeMem))
			case OpClosureLECall:
				v.Op = OpClosureCall
				rts := abi.RegisterTypes(v.Aux.(*AuxCall).abiInfo.OutParams())
				v.Type = types.NewResults(append(rts, types.TypeMem))
			case OpInterLECall:
				v.Op = OpInterCall
				rts := abi.RegisterTypes(v.Aux.(*AuxCall).abiInfo.OutParams())
				v.Type = types.NewResults(append(rts, types.TypeMem))
			}
		}
	}

	// Step 5: dedup OpArgXXXReg values. Mostly it is already dedup'd by commonArgs,
	// but there are cases that we have same OpArgXXXReg values with different types.
	// E.g. string is sometimes decomposed as { *int8, int }, sometimes as { unsafe.Pointer, uintptr }.
	// (Can we avoid that?)
	var IArg, FArg [32]*Value
	for _, v := range f.Entry.Values {
		switch v.Op {
		case OpArgIntReg:
			i := v.AuxInt
			if w := IArg[i]; w != nil {
				if w.Type.Size() != v.Type.Size() {
					f.Fatalf("incompatible OpArgIntReg [%d]: %s and %s", i, v.LongString(), w.LongString())
				}
				if w.Type.IsUnsafePtr() && !v.Type.IsUnsafePtr() {
					// Update unsafe.Pointer type if we know the actual pointer type.
					w.Type = v.Type
				}
				// TODO: don't dedup pointer and scalar? Rewrite to OpConvert? Can it happen?
				v.copyOf(w)
			} else {
				IArg[i] = v
			}
		case OpArgFloatReg:
			i := v.AuxInt
			if w := FArg[i]; w != nil {
				if w.Type.Size() != v.Type.Size() {
					f.Fatalf("incompatible OpArgFloatReg [%d]: %v and %v", i, v, w)
				}
				v.copyOf(w)
			} else {
				FArg[i] = v
			}
		}
	}

	// Step 6: elide any copies introduced.
	// Update named values.
	for _, name := range f.Names {
		values := f.NamedValues[*name]
		for i, v := range values {
			if v.Op == OpCopy {
				a := v.Args[0]
				for a.Op == OpCopy {
					a = a.Args[0]
				}
				values[i] = a
			}
		}
	}
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
					x.invalidateRecursively(a)
					a = b
				}
			}
		}
	}

	// Rewriting can attach lines to values that are unlikely to survive code generation, so move them to a use.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			for _, a := range v.Args {
				if a.Pos.IsStmt() != src.PosIsStmt {
					continue
				}
				if a.Type.IsMemory() {
					continue
				}
				if a.Pos.Line() != v.Pos.Line() {
					continue
				}
				if !a.Pos.SameFile(v.Pos) {
					continue
				}
				switch a.Op {
				case OpArgIntReg, OpArgFloatReg, OpSelectN:
					v.Pos = v.Pos.WithIsStmt()
					a.Pos = a.Pos.WithDefaultStmt()
				}
			}
		}
	}
}

// rewriteArgToMemOrRegs converts OpArg v in-place into the register version of v,
// if that is appropriate.
func (x *expandState) rewriteArgToMemOrRegs(v *Value) *Value {
	if x.debug > 1 {
		x.indent(3)
		defer x.indent(-3)
		x.Printf("rewriteArgToMemOrRegs(%s)\n", v.LongString())
	}
	pa := x.prAssignForArg(v)
	switch len(pa.Registers) {
	case 0:
		frameOff := v.Aux.(*ir.Name).FrameOffset()
		if pa.Offset() != int32(frameOff+x.f.ABISelf.LocalsOffset()) {
			panic(fmt.Errorf("Parameter assignment %d and OpArg.Aux frameOffset %d disagree, op=%s",
				pa.Offset(), frameOff, v.LongString()))
		}
	case 1:
		t := v.Type
		key := selKey{v, 0, t.Size(), t}
		w := x.commonArgs[key]
		if w != nil && w.Uses != 0 { // do not reuse dead value
			v.copyOf(w)
			break
		}
		r := pa.Registers[0]
		var i int64
		v.Op, i = ArgOpAndRegisterFor(r, x.f.ABISelf)
		v.Aux = &AuxNameOffset{v.Aux.(*ir.Name), 0}
		v.AuxInt = i
		x.commonArgs[key] = v

	default:
		panic(badVal("Saw unexpanded OpArg", v))
	}
	if x.debug > 1 {
		x.Printf("-->%s\n", v.LongString())
	}
	return v
}

// newArgToMemOrRegs either rewrites toReplace into an OpArg referencing memory or into an OpArgXXXReg to a register,
// or rewrites it into a copy of the appropriate OpArgXXX.  The actual OpArgXXX is determined by combining baseArg (an OpArg)
// with offset, regOffset, and t to determine which portion of it to reference (either all or a part, in memory or in registers).
func (x *expandState) newArgToMemOrRegs(baseArg, toReplace *Value, offset int64, regOffset Abi1RO, t *types.Type, pos src.XPos) *Value {
	if x.debug > 1 {
		x.indent(3)
		defer x.indent(-3)
		x.Printf("newArgToMemOrRegs(base=%s; toReplace=%s; t=%s; memOff=%d; regOff=%d)\n", baseArg.String(), toReplace.LongString(), t.String(), offset, regOffset)
	}
	key := selKey{baseArg, offset, t.Size(), t}
	w := x.commonArgs[key]
	if w != nil && w.Uses != 0 { // do not reuse dead value
		if toReplace != nil {
			toReplace.copyOf(w)
			if x.debug > 1 {
				x.Printf("...replace %s\n", toReplace.LongString())
			}
		}
		if x.debug > 1 {
			x.Printf("-->%s\n", w.LongString())
		}
		return w
	}

	pa := x.prAssignForArg(baseArg)
	if len(pa.Registers) == 0 { // Arg is on stack
		frameOff := baseArg.Aux.(*ir.Name).FrameOffset()
		if pa.Offset() != int32(frameOff+x.f.ABISelf.LocalsOffset()) {
			panic(fmt.Errorf("Parameter assignment %d and OpArg.Aux frameOffset %d disagree, op=%s",
				pa.Offset(), frameOff, baseArg.LongString()))
		}
		aux := baseArg.Aux
		auxInt := baseArg.AuxInt + offset
		if toReplace != nil && toReplace.Block == baseArg.Block {
			toReplace.reset(OpArg)
			toReplace.Aux = aux
			toReplace.AuxInt = auxInt
			toReplace.Type = t
			w = toReplace
		} else {
			w = baseArg.Block.NewValue0IA(baseArg.Pos, OpArg, t, auxInt, aux)
		}
		x.commonArgs[key] = w
		if toReplace != nil {
			toReplace.copyOf(w)
		}
		if x.debug > 1 {
			x.Printf("-->%s\n", w.LongString())
		}
		return w
	}
	// Arg is in registers
	r := pa.Registers[regOffset]
	op, auxInt := ArgOpAndRegisterFor(r, x.f.ABISelf)
	if op == OpArgIntReg && t.IsFloat() || op == OpArgFloatReg && t.IsInteger() {
		fmt.Printf("pa=%v\nx.f.OwnAux.abiInfo=%s\n",
			pa.ToString(x.f.ABISelf, true),
			x.f.OwnAux.abiInfo.String())
		panic(fmt.Errorf("Op/Type mismatch, op=%s, type=%s", op.String(), t.String()))
	}
	if baseArg.AuxInt != 0 {
		base.Fatalf("BaseArg %s bound to registers has non-zero AuxInt", baseArg.LongString())
	}
	aux := &AuxNameOffset{baseArg.Aux.(*ir.Name), offset}
	if toReplace != nil && toReplace.Block == baseArg.Block {
		toReplace.reset(op)
		toReplace.Aux = aux
		toReplace.AuxInt = auxInt
		toReplace.Type = t
		w = toReplace
	} else {
		w = baseArg.Block.NewValue0IA(baseArg.Pos, op, t, auxInt, aux)
	}
	x.commonArgs[key] = w
	if toReplace != nil {
		toReplace.copyOf(w)
	}
	if x.debug > 1 {
		x.Printf("-->%s\n", w.LongString())
	}
	return w

}

// argOpAndRegisterFor converts an abi register index into an ssa Op and corresponding
// arg register index.
func ArgOpAndRegisterFor(r abi.RegIndex, abiConfig *abi.ABIConfig) (Op, int64) {
	i := abiConfig.FloatIndexFor(r)
	if i >= 0 { // float PR
		return OpArgFloatReg, i
	}
	return OpArgIntReg, int64(r)
}
