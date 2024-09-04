// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"cmp"
	"slices"
)

// memcombine combines smaller loads and stores into larger ones.
// We ensure this generates good code for encoding/binary operations.
// It may help other cases also.
func memcombine(f *Func) {
	// This optimization requires that the architecture has
	// unaligned loads and unaligned stores.
	if !f.Config.unalignedOK {
		return
	}

	memcombineLoads(f)
	memcombineStores(f)
}

func memcombineLoads(f *Func) {
	// Find "OR trees" to start with.
	mark := f.newSparseSet(f.NumValues())
	defer f.retSparseSet(mark)
	var order []*Value

	// Mark all values that are the argument of an OR.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op == OpOr16 || v.Op == OpOr32 || v.Op == OpOr64 {
				mark.add(v.Args[0].ID)
				mark.add(v.Args[1].ID)
			}
		}
	}
	for _, b := range f.Blocks {
		order = order[:0]
		for _, v := range b.Values {
			if v.Op != OpOr16 && v.Op != OpOr32 && v.Op != OpOr64 {
				continue
			}
			if mark.contains(v.ID) {
				// marked - means it is not the root of an OR tree
				continue
			}
			// Add the OR tree rooted at v to the order.
			// We use BFS here, but any walk that puts roots before leaves would work.
			i := len(order)
			order = append(order, v)
			for ; i < len(order); i++ {
				x := order[i]
				for j := 0; j < 2; j++ {
					a := x.Args[j]
					if a.Op == OpOr16 || a.Op == OpOr32 || a.Op == OpOr64 {
						order = append(order, a)
					}
				}
			}
		}
		for _, v := range order {
			max := f.Config.RegSize
			switch v.Op {
			case OpOr64:
			case OpOr32:
				max = 4
			case OpOr16:
				max = 2
			default:
				continue
			}
			for n := max; n > 1; n /= 2 {
				if combineLoads(v, n) {
					break
				}
			}
		}
	}
}

// A BaseAddress represents the address ptr+idx, where
// ptr is a pointer type and idx is an integer type.
// idx may be nil, in which case it is treated as 0.
type BaseAddress struct {
	ptr *Value
	idx *Value
}

// splitPtr returns the base address of ptr and any
// constant offset from that base.
// BaseAddress{ptr,nil},0 is always a valid result, but splitPtr
// tries to peel away as many constants into off as possible.
func splitPtr(ptr *Value) (BaseAddress, int64) {
	var idx *Value
	var off int64
	for {
		if ptr.Op == OpOffPtr {
			off += ptr.AuxInt
			ptr = ptr.Args[0]
		} else if ptr.Op == OpAddPtr {
			if idx != nil {
				// We have two or more indexing values.
				// Pick the first one we found.
				return BaseAddress{ptr: ptr, idx: idx}, off
			}
			idx = ptr.Args[1]
			if idx.Op == OpAdd32 || idx.Op == OpAdd64 {
				if idx.Args[0].Op == OpConst32 || idx.Args[0].Op == OpConst64 {
					off += idx.Args[0].AuxInt
					idx = idx.Args[1]
				} else if idx.Args[1].Op == OpConst32 || idx.Args[1].Op == OpConst64 {
					off += idx.Args[1].AuxInt
					idx = idx.Args[0]
				}
			}
			ptr = ptr.Args[0]
		} else {
			return BaseAddress{ptr: ptr, idx: idx}, off
		}
	}
}

func combineLoads(root *Value, n int64) bool {
	orOp := root.Op
	var shiftOp Op
	switch orOp {
	case OpOr64:
		shiftOp = OpLsh64x64
	case OpOr32:
		shiftOp = OpLsh32x64
	case OpOr16:
		shiftOp = OpLsh16x64
	default:
		return false
	}

	// Find n values that are ORed together with the above op.
	a := make([]*Value, 0, 8)
	a = append(a, root)
	for i := 0; i < len(a) && int64(len(a)) < n; i++ {
		v := a[i]
		if v.Uses != 1 && v != root {
			// Something in this subtree is used somewhere else.
			return false
		}
		if v.Op == orOp {
			a[i] = v.Args[0]
			a = append(a, v.Args[1])
			i--
		}
	}
	if int64(len(a)) != n {
		return false
	}

	// Check that the first entry to see what ops we're looking for.
	// All the entries should be of the form shift(extend(load)), maybe with no shift.
	v := a[0]
	if v.Op == shiftOp {
		v = v.Args[0]
	}
	var extOp Op
	if orOp == OpOr64 && (v.Op == OpZeroExt8to64 || v.Op == OpZeroExt16to64 || v.Op == OpZeroExt32to64) ||
		orOp == OpOr32 && (v.Op == OpZeroExt8to32 || v.Op == OpZeroExt16to32) ||
		orOp == OpOr16 && v.Op == OpZeroExt8to16 {
		extOp = v.Op
		v = v.Args[0]
	} else {
		return false
	}
	if v.Op != OpLoad {
		return false
	}
	base, _ := splitPtr(v.Args[0])
	mem := v.Args[1]
	size := v.Type.Size()

	if root.Block.Func.Config.arch == "S390X" {
		// s390x can't handle unaligned accesses to global variables.
		if base.ptr.Op == OpAddr {
			return false
		}
	}

	// Check all the entries, extract useful info.
	type LoadRecord struct {
		load   *Value
		offset int64 // offset of load address from base
		shift  int64
	}
	r := make([]LoadRecord, n, 8)
	for i := int64(0); i < n; i++ {
		v := a[i]
		if v.Uses != 1 {
			return false
		}
		shift := int64(0)
		if v.Op == shiftOp {
			if v.Args[1].Op != OpConst64 {
				return false
			}
			shift = v.Args[1].AuxInt
			v = v.Args[0]
			if v.Uses != 1 {
				return false
			}
		}
		if v.Op != extOp {
			return false
		}
		load := v.Args[0]
		if load.Op != OpLoad {
			return false
		}
		if load.Uses != 1 {
			return false
		}
		if load.Args[1] != mem {
			return false
		}
		p, off := splitPtr(load.Args[0])
		if p != base {
			return false
		}
		r[i] = LoadRecord{load: load, offset: off, shift: shift}
	}

	// Sort in memory address order.
	slices.SortFunc(r, func(a, b LoadRecord) int {
		return cmp.Compare(a.offset, b.offset)
	})

	// Check that we have contiguous offsets.
	for i := int64(0); i < n; i++ {
		if r[i].offset != r[0].offset+i*size {
			return false
		}
	}

	// Check for reads in little-endian or big-endian order.
	shift0 := r[0].shift
	isLittleEndian := true
	for i := int64(0); i < n; i++ {
		if r[i].shift != shift0+i*size*8 {
			isLittleEndian = false
			break
		}
	}
	isBigEndian := true
	for i := int64(0); i < n; i++ {
		if r[i].shift != shift0-i*size*8 {
			isBigEndian = false
			break
		}
	}
	if !isLittleEndian && !isBigEndian {
		return false
	}

	// Find a place to put the new load.
	// This is tricky, because it has to be at a point where
	// its memory argument is live. We can't just put it in root.Block.
	// We use the block of the latest load.
	loads := make([]*Value, n, 8)
	for i := int64(0); i < n; i++ {
		loads[i] = r[i].load
	}
	loadBlock := mergePoint(root.Block, loads...)
	if loadBlock == nil {
		return false
	}
	// Find a source position to use.
	pos := src.NoXPos
	for _, load := range loads {
		if load.Block == loadBlock {
			pos = load.Pos
			break
		}
	}
	if pos == src.NoXPos {
		return false
	}

	// Check to see if we need byte swap before storing.
	needSwap := isLittleEndian && root.Block.Func.Config.BigEndian ||
		isBigEndian && !root.Block.Func.Config.BigEndian
	if needSwap && (size != 1 || !root.Block.Func.Config.haveByteSwap(n)) {
		return false
	}

	// This is the commit point.

	// First, issue load at lowest address.
	v = loadBlock.NewValue2(pos, OpLoad, sizeType(n*size), r[0].load.Args[0], mem)

	// Byte swap if needed,
	if needSwap {
		v = byteSwap(loadBlock, pos, v)
	}

	// Extend if needed.
	if n*size < root.Type.Size() {
		v = zeroExtend(loadBlock, pos, v, n*size, root.Type.Size())
	}

	// Shift if needed.
	if isLittleEndian && shift0 != 0 {
		v = leftShift(loadBlock, pos, v, shift0)
	}
	if isBigEndian && shift0-(n-1)*size*8 != 0 {
		v = leftShift(loadBlock, pos, v, shift0-(n-1)*size*8)
	}

	// Install with (Copy v).
	root.reset(OpCopy)
	root.AddArg(v)

	// Clobber the loads, just to prevent additional work being done on
	// subtrees (which are now unreachable).
	for i := int64(0); i < n; i++ {
		clobber(r[i].load)
	}
	return true
}

func memcombineStores(f *Func) {
	mark := f.newSparseSet(f.NumValues())
	defer f.retSparseSet(mark)
	var order []*Value

	for _, b := range f.Blocks {
		// Mark all stores which are not last in a store sequence.
		mark.clear()
		for _, v := range b.Values {
			if v.Op == OpStore {
				mark.add(v.MemoryArg().ID)
			}
		}

		// pick an order for visiting stores such that
		// later stores come earlier in the ordering.
		order = order[:0]
		for _, v := range b.Values {
			if v.Op != OpStore {
				continue
			}
			if mark.contains(v.ID) {
				continue // not last in a chain of stores
			}
			for {
				order = append(order, v)
				v = v.Args[2]
				if v.Block != b || v.Op != OpStore {
					break
				}
			}
		}

		// Look for combining opportunities at each store in queue order.
		for _, v := range order {
			if v.Op != OpStore { // already rewritten
				continue
			}

			size := v.Aux.(*types.Type).Size()
			if size >= f.Config.RegSize || size == 0 {
				continue
			}

			for n := f.Config.RegSize / size; n > 1; n /= 2 {
				if combineStores(v, n) {
					continue
				}
			}
		}
	}
}

// Try to combine the n stores ending in root.
// Returns true if successful.
func combineStores(root *Value, n int64) bool {
	// Helper functions.
	type StoreRecord struct {
		store  *Value
		offset int64
	}
	getShiftBase := func(a []StoreRecord) *Value {
		x := a[0].store.Args[1]
		y := a[1].store.Args[1]
		switch x.Op {
		case OpTrunc64to8, OpTrunc64to16, OpTrunc64to32, OpTrunc32to8, OpTrunc32to16, OpTrunc16to8:
			x = x.Args[0]
		default:
			return nil
		}
		switch y.Op {
		case OpTrunc64to8, OpTrunc64to16, OpTrunc64to32, OpTrunc32to8, OpTrunc32to16, OpTrunc16to8:
			y = y.Args[0]
		default:
			return nil
		}
		var x2 *Value
		switch x.Op {
		case OpRsh64Ux64, OpRsh32Ux64, OpRsh16Ux64:
			x2 = x.Args[0]
		default:
		}
		var y2 *Value
		switch y.Op {
		case OpRsh64Ux64, OpRsh32Ux64, OpRsh16Ux64:
			y2 = y.Args[0]
		default:
		}
		if y2 == x {
			// a shift of x and x itself.
			return x
		}
		if x2 == y {
			// a shift of y and y itself.
			return y
		}
		if x2 == y2 {
			// 2 shifts both of the same argument.
			return x2
		}
		return nil
	}
	isShiftBase := func(v, base *Value) bool {
		val := v.Args[1]
		switch val.Op {
		case OpTrunc64to8, OpTrunc64to16, OpTrunc64to32, OpTrunc32to8, OpTrunc32to16, OpTrunc16to8:
			val = val.Args[0]
		default:
			return false
		}
		if val == base {
			return true
		}
		switch val.Op {
		case OpRsh64Ux64, OpRsh32Ux64, OpRsh16Ux64:
			val = val.Args[0]
		default:
			return false
		}
		return val == base
	}
	shift := func(v, base *Value) int64 {
		val := v.Args[1]
		switch val.Op {
		case OpTrunc64to8, OpTrunc64to16, OpTrunc64to32, OpTrunc32to8, OpTrunc32to16, OpTrunc16to8:
			val = val.Args[0]
		default:
			return -1
		}
		if val == base {
			return 0
		}
		switch val.Op {
		case OpRsh64Ux64, OpRsh32Ux64, OpRsh16Ux64:
			val = val.Args[1]
		default:
			return -1
		}
		if val.Op != OpConst64 {
			return -1
		}
		return val.AuxInt
	}

	// Element size of the individual stores.
	size := root.Aux.(*types.Type).Size()
	if size*n > root.Block.Func.Config.RegSize {
		return false
	}

	// Gather n stores to look at. Check easy conditions we require.
	a := make([]StoreRecord, 0, 8)
	rbase, roff := splitPtr(root.Args[0])
	if root.Block.Func.Config.arch == "S390X" {
		// s390x can't handle unaligned accesses to global variables.
		if rbase.ptr.Op == OpAddr {
			return false
		}
	}
	a = append(a, StoreRecord{root, roff})
	for i, x := int64(1), root.Args[2]; i < n; i, x = i+1, x.Args[2] {
		if x.Op != OpStore {
			return false
		}
		if x.Block != root.Block {
			return false
		}
		if x.Uses != 1 { // Note: root can have more than one use.
			return false
		}
		if x.Aux.(*types.Type).Size() != size {
			// TODO: the constant source and consecutive load source cases
			// do not need all the stores to be the same size.
			return false
		}
		base, off := splitPtr(x.Args[0])
		if base != rbase {
			return false
		}
		a = append(a, StoreRecord{x, off})
	}
	// Before we sort, grab the memory arg the result should have.
	mem := a[n-1].store.Args[2]
	// Also grab position of first store (last in array = first in memory order).
	pos := a[n-1].store.Pos

	// Sort stores in increasing address order.
	slices.SortFunc(a, func(a, b StoreRecord) int {
		return cmp.Compare(a.offset, b.offset)
	})

	// Check that everything is written to sequential locations.
	for i := int64(0); i < n; i++ {
		if a[i].offset != a[0].offset+i*size {
			return false
		}
	}

	// Memory location we're going to write at (the lowest one).
	ptr := a[0].store.Args[0]

	// Check for constant stores
	isConst := true
	for i := int64(0); i < n; i++ {
		switch a[i].store.Args[1].Op {
		case OpConst32, OpConst16, OpConst8, OpConstBool:
		default:
			isConst = false
			break
		}
	}
	if isConst {
		// Modify root to do all the stores.
		var c int64
		mask := int64(1)<<(8*size) - 1
		for i := int64(0); i < n; i++ {
			s := 8 * size * int64(i)
			if root.Block.Func.Config.BigEndian {
				s = 8*size*(n-1) - s
			}
			c |= (a[i].store.Args[1].AuxInt & mask) << s
		}
		var cv *Value
		switch size * n {
		case 2:
			cv = root.Block.Func.ConstInt16(types.Types[types.TUINT16], int16(c))
		case 4:
			cv = root.Block.Func.ConstInt32(types.Types[types.TUINT32], int32(c))
		case 8:
			cv = root.Block.Func.ConstInt64(types.Types[types.TUINT64], c)
		}

		// Move all the stores to the root.
		for i := int64(0); i < n; i++ {
			v := a[i].store
			if v == root {
				v.Aux = cv.Type // widen store type
				v.Pos = pos
				v.SetArg(0, ptr)
				v.SetArg(1, cv)
				v.SetArg(2, mem)
			} else {
				clobber(v)
				v.Type = types.Types[types.TBOOL] // erase memory type
			}
		}
		return true
	}

	// Check for consecutive loads as the source of the stores.
	var loadMem *Value
	var loadBase BaseAddress
	var loadIdx int64
	for i := int64(0); i < n; i++ {
		load := a[i].store.Args[1]
		if load.Op != OpLoad {
			loadMem = nil
			break
		}
		if load.Uses != 1 {
			loadMem = nil
			break
		}
		if load.Type.IsPtr() {
			// Don't combine stores containing a pointer, as we need
			// a write barrier for those. This can't currently happen,
			// but might in the future if we ever have another
			// 8-byte-reg/4-byte-ptr architecture like amd64p32.
			loadMem = nil
			break
		}
		mem := load.Args[1]
		base, idx := splitPtr(load.Args[0])
		if loadMem == nil {
			// First one we found
			loadMem = mem
			loadBase = base
			loadIdx = idx
			continue
		}
		if base != loadBase || mem != loadMem {
			loadMem = nil
			break
		}
		if idx != loadIdx+(a[i].offset-a[0].offset) {
			loadMem = nil
			break
		}
	}
	if loadMem != nil {
		// Modify the first load to do a larger load instead.
		load := a[0].store.Args[1]
		switch size * n {
		case 2:
			load.Type = types.Types[types.TUINT16]
		case 4:
			load.Type = types.Types[types.TUINT32]
		case 8:
			load.Type = types.Types[types.TUINT64]
		}

		// Modify root to do the store.
		for i := int64(0); i < n; i++ {
			v := a[i].store
			if v == root {
				v.Aux = load.Type // widen store type
				v.Pos = pos
				v.SetArg(0, ptr)
				v.SetArg(1, load)
				v.SetArg(2, mem)
			} else {
				clobber(v)
				v.Type = types.Types[types.TBOOL] // erase memory type
			}
		}
		return true
	}

	// Check that all the shift/trunc are of the same base value.
	shiftBase := getShiftBase(a)
	if shiftBase == nil {
		return false
	}
	for i := int64(0); i < n; i++ {
		if !isShiftBase(a[i].store, shiftBase) {
			return false
		}
	}

	// Check for writes in little-endian or big-endian order.
	isLittleEndian := true
	shift0 := shift(a[0].store, shiftBase)
	for i := int64(1); i < n; i++ {
		if shift(a[i].store, shiftBase) != shift0+i*size*8 {
			isLittleEndian = false
			break
		}
	}
	isBigEndian := true
	for i := int64(1); i < n; i++ {
		if shift(a[i].store, shiftBase) != shift0-i*size*8 {
			isBigEndian = false
			break
		}
	}
	if !isLittleEndian && !isBigEndian {
		return false
	}

	// Check to see if we need byte swap before storing.
	needSwap := isLittleEndian && root.Block.Func.Config.BigEndian ||
		isBigEndian && !root.Block.Func.Config.BigEndian
	if needSwap && (size != 1 || !root.Block.Func.Config.haveByteSwap(n)) {
		return false
	}

	// This is the commit point.

	// Modify root to do all the stores.
	sv := shiftBase
	if isLittleEndian && shift0 != 0 {
		sv = rightShift(root.Block, root.Pos, sv, shift0)
	}
	if isBigEndian && shift0-(n-1)*size*8 != 0 {
		sv = rightShift(root.Block, root.Pos, sv, shift0-(n-1)*size*8)
	}
	if sv.Type.Size() > size*n {
		sv = truncate(root.Block, root.Pos, sv, sv.Type.Size(), size*n)
	}
	if needSwap {
		sv = byteSwap(root.Block, root.Pos, sv)
	}

	// Move all the stores to the root.
	for i := int64(0); i < n; i++ {
		v := a[i].store
		if v == root {
			v.Aux = sv.Type // widen store type
			v.Pos = pos
			v.SetArg(0, ptr)
			v.SetArg(1, sv)
			v.SetArg(2, mem)
		} else {
			clobber(v)
			v.Type = types.Types[types.TBOOL] // erase memory type
		}
	}
	return true
}

func sizeType(size int64) *types.Type {
	switch size {
	case 8:
		return types.Types[types.TUINT64]
	case 4:
		return types.Types[types.TUINT32]
	case 2:
		return types.Types[types.TUINT16]
	default:
		base.Fatalf("bad size %d\n", size)
		return nil
	}
}

func truncate(b *Block, pos src.XPos, v *Value, from, to int64) *Value {
	switch from*10 + to {
	case 82:
		return b.NewValue1(pos, OpTrunc64to16, types.Types[types.TUINT16], v)
	case 84:
		return b.NewValue1(pos, OpTrunc64to32, types.Types[types.TUINT32], v)
	case 42:
		return b.NewValue1(pos, OpTrunc32to16, types.Types[types.TUINT16], v)
	default:
		base.Fatalf("bad sizes %d %d\n", from, to)
		return nil
	}
}
func zeroExtend(b *Block, pos src.XPos, v *Value, from, to int64) *Value {
	switch from*10 + to {
	case 24:
		return b.NewValue1(pos, OpZeroExt16to32, types.Types[types.TUINT32], v)
	case 28:
		return b.NewValue1(pos, OpZeroExt16to64, types.Types[types.TUINT64], v)
	case 48:
		return b.NewValue1(pos, OpZeroExt32to64, types.Types[types.TUINT64], v)
	default:
		base.Fatalf("bad sizes %d %d\n", from, to)
		return nil
	}
}

func leftShift(b *Block, pos src.XPos, v *Value, shift int64) *Value {
	s := b.Func.ConstInt64(types.Types[types.TUINT64], shift)
	size := v.Type.Size()
	switch size {
	case 8:
		return b.NewValue2(pos, OpLsh64x64, v.Type, v, s)
	case 4:
		return b.NewValue2(pos, OpLsh32x64, v.Type, v, s)
	case 2:
		return b.NewValue2(pos, OpLsh16x64, v.Type, v, s)
	default:
		base.Fatalf("bad size %d\n", size)
		return nil
	}
}
func rightShift(b *Block, pos src.XPos, v *Value, shift int64) *Value {
	s := b.Func.ConstInt64(types.Types[types.TUINT64], shift)
	size := v.Type.Size()
	switch size {
	case 8:
		return b.NewValue2(pos, OpRsh64Ux64, v.Type, v, s)
	case 4:
		return b.NewValue2(pos, OpRsh32Ux64, v.Type, v, s)
	case 2:
		return b.NewValue2(pos, OpRsh16Ux64, v.Type, v, s)
	default:
		base.Fatalf("bad size %d\n", size)
		return nil
	}
}
func byteSwap(b *Block, pos src.XPos, v *Value) *Value {
	switch v.Type.Size() {
	case 8:
		return b.NewValue1(pos, OpBswap64, v.Type, v)
	case 4:
		return b.NewValue1(pos, OpBswap32, v.Type, v)
	case 2:
		return b.NewValue1(pos, OpBswap16, v.Type, v)

	default:
		v.Fatalf("bad size %d\n", v.Type.Size())
		return nil
	}
}
