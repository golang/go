// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"internal/buildcfg"
)

type autovecLane struct {
	size, shift                 int64
	insert, broadcast           Op
	add, sub, mul, and, or, xor Op
	left, right, signedRight    Op
}

var autovecLanes = [...]autovecLane{
	{1, 0, OpSetElemInt8x16, Opbroadcast1To16Int8x16,
		OpAddInt8x16, OpSubInt8x16, OpInvalid, OpAndInt8x16, OpOrInt8x16, OpXorInt8x16,
		OpInvalid, OpInvalid, OpInvalid},
	{2, 1, OpSetElemInt16x8, Opbroadcast1To8Int16x8,
		OpAddInt16x8, OpSubInt16x8, OpMulInt16x8, OpAndInt16x8, OpOrInt16x8, OpXorInt16x8,
		OpShiftAllLeftInt16x8, OpShiftAllRightUint16x8, OpShiftAllRightInt16x8},
	{4, 2, OpSetElemInt32x4, Opbroadcast1To4Int32x4,
		OpAddInt32x4, OpSubInt32x4, OpMulInt32x4, OpAndInt32x4, OpOrInt32x4, OpXorInt32x4,
		OpShiftAllLeftInt32x4, OpShiftAllRightUint32x4, OpShiftAllRightInt32x4},
	{8, 3, OpSetElemInt64x2, Opbroadcast1To2Int64x2,
		OpAddInt64x2, OpSubInt64x2, OpInvalid, OpAndInt64x2, OpOrInt64x2, OpXorInt64x2,
		OpShiftAllLeftInt64x2, OpShiftAllRightUint64x2, OpInvalid},
}

// An autovecLoop is complete before the CFG changes. Every skipped scalar
// value belongs to its index, address, or expression graph, and only the
// index and memory phis may be observed outside the loop.
type autovecLoop struct {
	header, body *Block
	predecessor  Edge
	entryIndex   int
	index, mem   *Value
	limit, store *Value
	lane         autovecLane
	ops          map[*Value]Op
	bases        map[*Value]*Value
	sources      []*Value
	covered      map[*Value]bool
	dom          SparseTree
}

func autovec(f *Func) {
	if !buildcfg.Experiment.SIMD || !f.Config.optimize || f.NoSplit || f.Config.Race ||
		base.Flag.Race || base.Flag.MSan || base.Flag.ASan || f.Config.SoftFloat {
		return
	}
	if f.Config.arch != "arm64" && (f.Config.arch != "amd64" || buildcfg.GOAMD64 < 3) {
		return
	}
	// New blocks belong to vector prefixes. Revisit only the original headers, and
	// obtain fresh dominance information after each accepted rewrite.
	for _, h := range f.Blocks[:len(f.Blocks)] {
		p := autovecMatch(f, h)
		if p == nil {
			continue
		}
		p.rewrite()
		if f.pass.debug > 0 {
			f.Warnl(p.store.Pos, "autovectorized %d-bit integer loop with %d lanes", p.lane.size*8, 16/p.lane.size)
		}
	}
}

func autovecMatch(f *Func, h *Block) *autovecLoop {
	if h.Kind != BlockIf || len(h.Preds) != 2 || h.Controls[0].Op != OpLess64 {
		return nil
	}
	index, limit := h.Controls[0].Args[0], h.Controls[0].Args[1]
	if index.Op != OpPhi || index.Block != h || len(index.Args) != 2 || !index.Type.IsInteger() || index.Type.Size() != 8 {
		return nil
	}
	b := h.Succs[0].b
	if b == h || b.Kind != BlockPlain || len(b.Preds) != 1 || b.Succs[0].b != h || h.Succs[1].b == h {
		return nil
	}
	back := b.Succs[0].i
	entry := 1 - back
	start, next := index.Args[entry], index.Args[back]
	if next.Op != OpAdd64 || next.Block != b {
		return nil
	}
	one := next.Args[1]
	if next.Args[0] != index {
		if next.Args[1] != index {
			return nil
		}
		one = next.Args[0]
	}
	if one.Op != OpConst64 || one.AuxInt != 1 {
		return nil
	}
	var mem *Value
	for _, v := range h.Values {
		if v.Op == OpPhi && v.Type.IsMemory() {
			if mem != nil {
				return nil
			}
			mem = v
		}
	}
	if mem == nil || len(mem.Args) != 2 {
		return nil
	}
	store := mem.Args[back]
	if store.Op != OpStore || store.Block != b || store.Args[2] != mem {
		return nil
	}
	t := store.Aux.(*types.Type)
	if !t.IsInteger() {
		return nil
	}
	var lane autovecLane
	for _, l := range autovecLanes {
		if l.size == t.Size() {
			lane = l
			break
		}
	}
	if lane.size == 0 {
		return nil
	}
	p := &autovecLoop{
		header: h, body: b, predecessor: h.Preds[entry], entryIndex: entry,
		index: index, mem: mem, limit: limit, store: store, lane: lane,
		ops: make(map[*Value]Op), bases: make(map[*Value]*Value),
		covered: make(map[*Value]bool), dom: f.Sdom(),
	}
	if p.inside(p.predecessor.b) || !p.invariant(start) || !p.invariant(limit) || !p.invariant(mem.Args[entry]) {
		return nil
	}
	for _, v := range []*Value{index, mem, h.Controls[0], next, one} {
		p.covered[v] = true
	}
	if !p.address(store.Args[0]) || !p.expression(store.Args[1]) {
		return nil
	}
	p.covered[store] = true
	for _, block := range []*Block{h, b} {
		for _, v := range block.Values {
			if !p.covered[v] && !v.isGenericIntConst() {
				return nil
			}
		}
	}
	for _, block := range f.Blocks {
		if p.inside(block) {
			continue
		}
		for _, v := range block.Values {
			for _, a := range v.Args {
				if p.inside(a.Block) && a != index && a != mem {
					return nil
				}
			}
		}
		for _, v := range block.ControlValues() {
			if p.inside(v.Block) && v != index && v != mem {
				return nil
			}
		}
	}
	return p
}

func (p *autovecLoop) inside(b *Block) bool {
	return b == p.header || b == p.body
}

func (p *autovecLoop) invariant(v *Value) bool {
	return !p.inside(v.Block) && p.dom.IsAncestorEq(v.Block, p.predecessor.b)
}

func (p *autovecLoop) address(v *Value) bool {
	if _, ok := p.bases[v]; ok {
		return true
	}
	if v.Op != OpAddPtr || !p.invariant(v.Args[0]) || !v.Args[0].Type.IsPtrShaped() {
		return false
	}
	offset := v.Args[1]
	if p.lane.shift == 0 {
		if offset != p.index {
			return false
		}
	} else {
		if offset.Op != OpLsh64x64 || offset.Args[0] != p.index ||
			offset.Args[1].Op != OpConst64 || offset.Args[1].AuxInt != p.lane.shift {
			return false
		}
		p.covered[offset.Args[1]] = true
	}
	p.covered[v], p.covered[offset] = true, true
	p.bases[v] = v.Args[0]
	return true
}

func (p *autovecLoop) expression(v *Value) bool {
	if _, ok := p.ops[v]; ok {
		return true
	}
	if !v.Type.IsInteger() || v.Type.Size() != p.lane.size {
		return false
	}
	if v.isGenericIntConst() || p.invariant(v) {
		p.ops[v] = OpInvalid // An already evaluated scalar, broadcast before the loop.
		p.covered[v] = true
		return true
	}
	if v.Block != p.body {
		return false
	}
	if v.Op == OpLoad {
		if v.Args[1] != p.mem || !p.address(v.Args[0]) {
			return false
		}
		p.ops[v], p.covered[v] = OpLoad, true
		base := p.bases[v.Args[0]]
		for _, source := range p.sources {
			if source == base {
				return true
			}
		}
		p.sources = append(p.sources, base)
		return true
	}
	var op Op
	scalarCount := false
	switch v.Op {
	case OpNeg8, OpNeg16, OpNeg32, OpNeg64:
		op = p.lane.sub
	case OpCom8, OpCom16, OpCom32, OpCom64:
		op = p.lane.xor
	case OpLsh8x64:
		// generic opt canonicalizes byte doubling to a shift.
		if v.Args[1].Op != OpConst64 || v.Args[1].AuxInt != 1 {
			return false
		}
		op, scalarCount = p.lane.add, true
	case OpLsh16x64, OpLsh32x64, OpLsh64x64:
		op, scalarCount = p.lane.left, true
	case OpRsh16Ux64, OpRsh32Ux64, OpRsh64Ux64:
		op, scalarCount = p.lane.right, true
	case OpRsh16x64, OpRsh32x64:
		op, scalarCount = p.lane.signedRight, true
	case OpAdd8, OpAdd16, OpAdd32, OpAdd64:
		op = p.lane.add
	case OpSub8, OpSub16, OpSub32, OpSub64:
		op = p.lane.sub
	case OpMul16, OpMul32:
		op = p.lane.mul
	case OpAnd8, OpAnd16, OpAnd32, OpAnd64:
		op = p.lane.and
	case OpOr8, OpOr16, OpOr32, OpOr64:
		op = p.lane.or
	case OpXor8, OpXor16, OpXor32, OpXor64:
		op = p.lane.xor
	}
	if op == OpInvalid || !p.expression(v.Args[0]) {
		return false
	}
	if scalarCount {
		count := v.Args[1]
		if count.Op != OpConst64 || uint64(count.AuxInt) >= uint64(p.lane.size*8) {
			return false
		}
		p.covered[count] = true
	} else if len(v.Args) == 2 && !p.expression(v.Args[1]) {
		return false
	}
	p.ops[v], p.covered[v] = op, true
	return true
}

func (p *autovecLoop) rewrite() {
	f := p.header.Func
	typ, pos := &f.Config.Types, p.store.Pos
	lanes := 16 / p.lane.size
	// Keep the call on the loop-entry edge so other successors retain their
	// memory. A separate setup block keeps broadcasts after the call.
	callBlock, pre := f.NewBlock(BlockPlain), f.NewBlock(BlockPlain)
	callBlock.Pos, pre.Pos = p.header.Pos, p.header.Pos
	aux := StaticAuxCall(f.fe.Syslook("logosAutovecPanicOnFault"), f.ABIDefault.ABIAnalyzeTypes(nil, []*types.Type{typ.Bool}))
	// Call expansion has already run. Only register results appear in the
	// call tuple; stack results use the returned memory.
	callType, memIndex := types.TypeResultMem, int64(0)
	if len(aux.RegsOfResult(0)) != 0 {
		callType = types.NewResults([]*types.Type{typ.Bool, types.TypeMem})
		memIndex = 1
	}
	call := callBlock.NewValue1A(pos, OpStaticCall, callType, aux, p.mem.Args[p.entryIndex])
	call.AuxInt = aux.ArgWidth()
	initialMem := callBlock.NewValue1I(pos, OpSelectN, types.TypeMem, memIndex, call)
	var paniconfault *Value
	if memIndex != 0 {
		paniconfault = callBlock.NewValue1I(pos, OpSelectN, typ.Bool, 0, call)
	} else {
		sp, _ := f.spSb()
		addr := f.ConstOffPtrSP(typ.Bool.PtrTo(), aux.OffsetOfResult(0), sp)
		paniconfault = pre.NewValue2(pos, OpLoad, typ.Bool, addr, initialMem)
	}
	safe := pre.NewValue1(pos, OpNot, typ.Bool, paniconfault)
	zero := pre.NewValue0I(pos, OpConst64, p.index.Type, 0)
	mask := pre.NewValue0I(pos, OpConst64, p.index.Type, -lanes)
	start := p.index.Args[p.entryIndex]
	var count, end *Value
	if start.Op == OpConst64 && start.AuxInt == 0 {
		// Keep the original zero-start path, including its simpler guard.
		count = pre.NewValue2(pos, OpAnd64, p.index.Type, p.limit, mask)
		end = count
	} else {
		active := pre.NewValue2(pos, OpLess64, typ.Bool, start, p.limit)
		difference := pre.NewValue2(pos, OpSub64, p.index.Type, p.limit, start)
		rounded := pre.NewValue2(pos, OpAnd64, p.index.Type, difference, mask)
		count = pre.NewValue3(pos, OpCondSelect, p.index.Type, rounded, zero, active)
		end = pre.NewValue2(pos, OpAdd64, p.index.Type, start, count)
		// For start < limit, unsigned modular subtraction equals their exact
		// nonnegative mathematical distance, even across signed overflow.
		// Rounding down makes start+count lie in [start, limit]. For an
		// inactive loop count is zero, so its vector loop cannot execute.
	}
	dst := p.bases[p.store.Args[0]]
	var dstInt *Value
	for _, source := range p.sources {
		if isSamePtr(dst, source) {
			continue
		}
		if dstInt == nil {
			dstInt = pre.NewValue2(pos, OpConvert, typ.Uintptr, dst, initialMem)
		}
		srcInt := pre.NewValue2(pos, OpConvert, typ.Uintptr, source, initialMem)
		before := pre.NewValue2(pos, OpLess64U, typ.Bool, dstInt, srcInt)
		forward := pre.NewValue2(pos, OpSub64, typ.Uintptr, srcInt, dstInt)
		backward := pre.NewValue2(pos, OpSub64, typ.Uintptr, dstInt, srcInt)
		distance := pre.NewValue3(pos, OpCondSelect, typ.Uintptr, forward, backward, before)
		if p.lane.shift != 0 {
			shift := pre.NewValue0I(pos, OpConst64, typ.UInt64, p.lane.shift)
			distance = pre.NewValue2I(pos, OpRsh64Ux64, typ.Uintptr, 1, distance, shift)
		}
		// Dividing the ordered distance avoids overflowing an end pointer or
		// a trip-count multiplication. Equal bases are safe in-place maps.
		disjoint := pre.NewValue2(pos, OpLeq64U, typ.Bool, count, distance)
		equal := pre.NewValue2(pos, OpEqPtr, typ.Bool, dst, source)
		guard := pre.NewValue2(pos, OpOrB, typ.Bool, equal, disjoint)
		safe = pre.NewValue2(pos, OpAndB, typ.Bool, safe, guard)
	}
	end = pre.NewValue3(pos, OpCondSelect, p.index.Type, end, start, safe)
	h, b := f.NewBlock(BlockIf), f.NewBlock(BlockPlain)
	h.Pos, b.Pos = p.header.Pos, p.body.Pos
	h.Likely = BranchLikely
	// Preserve the scalar header's predecessor index and its backedge.
	p.predecessor.b.Succs[p.predecessor.i] = Edge{callBlock, 0}
	callBlock.Preds = append(callBlock.Preds, p.predecessor)
	callBlock.AddEdgeTo(pre)
	pre.AddEdgeTo(h)
	h.AddEdgeTo(b)
	h.Succs = append(h.Succs, Edge{p.header, p.entryIndex})
	p.header.Preds[p.entryIndex] = Edge{h, 1}
	b.AddEdgeTo(h)
	index := h.NewValue1(pos, OpPhi, p.index.Type, start)
	mem := h.NewValue1(pos, OpPhi, types.TypeMem, initialMem)
	h.SetControl(h.NewValue2(pos, OpLess64, typ.Bool, index, end))
	values := make(map[*Value]*Value, len(p.ops))
	addresses := make(map[*Value]*Value, len(p.bases))
	offset := index
	if p.lane.shift != 0 {
		shift := pre.NewValue0I(pos, OpConst64, typ.UInt64, p.lane.shift)
		offset = b.NewValue2I(pos, OpLsh64x64, p.index.Type, 1, index, shift)
	}
	address := func(v *Value) *Value {
		if ptr := addresses[v]; ptr != nil {
			return ptr
		}
		ptr := b.NewValue2(v.Pos, OpAddPtr, v.Type, p.bases[v], offset)
		addresses[v] = ptr
		return ptr
	}
	broadcast := func(v *Value) *Value {
		scalar := v
		if p.inside(v.Block) {
			scalar = pre.NewValue0I(v.Pos, v.Op, v.Type, v.AuxInt)
		}
		z := pre.NewValue0(v.Pos, OpZeroSIMD, typ.Vec128)
		first := pre.NewValue2I(v.Pos, p.lane.insert, typ.Vec128, 0, z, scalar)
		return pre.NewValue1(v.Pos, p.lane.broadcast, typ.Vec128, first)
	}
	var expression func(*Value) *Value
	expression = func(v *Value) *Value {
		if result := values[v]; result != nil {
			return result
		}
		var result *Value
		switch op := p.ops[v]; op {
		case OpInvalid:
			result = broadcast(v)
		case OpLoad:
			result = b.NewValue2(v.Pos, OpLoad, typ.Vec128, address(v.Args[0]), mem)
		default:
			x := expression(v.Args[0])
			switch v.Op {
			case OpNeg8, OpNeg16, OpNeg32, OpNeg64:
				zero := pre.NewValue0(v.Pos, OpZeroSIMD, typ.Vec128)
				result = b.NewValue2(v.Pos, op, typ.Vec128, zero, x)
			case OpCom8, OpCom16, OpCom32, OpCom64:
				constant := [...]Op{OpConst8, OpConst16, OpConst32, OpConst64}[p.lane.shift]
				ones := pre.NewValue0I(v.Pos, constant, v.Type, -1)
				result = b.NewValue2(v.Pos, op, typ.Vec128, x, broadcast(ones))
			case OpLsh8x64:
				result = b.NewValue2(v.Pos, op, typ.Vec128, x, x)
			case OpLsh16x64, OpLsh32x64, OpLsh64x64,
				OpRsh16Ux64, OpRsh32Ux64, OpRsh64Ux64, OpRsh16x64, OpRsh32x64:
				count := v.Args[1]
				if p.inside(count.Block) {
					count = pre.NewValue0I(count.Pos, count.Op, count.Type, count.AuxInt)
				}
				result = b.NewValue2(v.Pos, op, typ.Vec128, x, count)
			default:
				result = b.NewValue2(v.Pos, op, typ.Vec128, x, expression(v.Args[1]))
			}
		}
		values[v] = result
		return result
	}
	result := expression(p.store.Args[1])
	store := b.NewValue3A(pos, OpStore, types.TypeMem, typ.Vec128, address(p.store.Args[0]), result, mem)
	step := pre.NewValue0I(pos, OpConst64, p.index.Type, lanes)
	index.AddArg(b.NewValue2(pos, OpAdd64, p.index.Type, index, step))
	mem.AddArg(store)
	p.index.SetArg(p.entryIndex, index)
	p.mem.SetArg(p.entryIndex, mem)
	f.invalidateCFG()
}
