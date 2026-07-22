// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacore

import (
	"fmt"

	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/objabi"
)

// IsNewObject reports whether v is a pointer to a freshly allocated & zeroed object,
// if so, also returns the memory state mem at which v is zero.
func IsNewObject(v *Value, select1 []*Value) (mem *Value, ok bool) {
	f := v.Block.Func
	c := f.Config
	if f.ABIDefault == f.ABI1 && len(c.IntParamRegs) >= 1 {
		if v.Op != ssaop.OpSelectN || v.AuxInt != 0 {
			return nil, false
		}
		mem = select1[v.Args[0].ID]
		if mem == nil {
			return nil, false
		}
	} else {
		if v.Op != ssaop.OpLoad {
			return nil, false
		}
		mem = v.MemoryArg()
		if mem.Op != ssaop.OpSelectN {
			return nil, false
		}
		if mem.Type != types.TypeMem {
			return nil, false
		} // assume it is the right selection if true
	}
	call := mem.Args[0]
	if call.Op != ssaop.OpStaticCall {
		return nil, false
	}
	// Check for new object, or for new object calls that have been transformed into size-specialized malloc calls.
	// Calls that have return type unsafe pointer may have originally been produced by flushPendingHeapAllocations
	// in the ssa generator, so may have not originally been newObject calls.
	var numParameters int64
	switch {
	case IsNewObjectCall(call.Aux):
		numParameters = 1
	case IsSpecializedMalloc(call.Aux) && !v.Type.IsUnsafePtr():
		numParameters = 3
	default:
		return nil, false
	}
	if f.ABIDefault == f.ABI1 && len(c.IntParamRegs) >= 1 {
		if v.Args[0] == call {
			return mem, true
		}
		return nil, false
	}
	if v.Args[0].Op != ssaop.OpOffPtr {
		return nil, false
	}
	if v.Args[0].Args[0].Op != ssaop.OpSP {
		return nil, false
	}
	if v.Args[0].AuxInt != c.Ctxt.Arch.FixedFrameSize+numParameters*c.RegSize { // offset of return value
		return nil, false
	}
	return mem, true
}

// A ZeroRegion records parts of an object which are known to be zero.
// A ZeroRegion only applies to a single memory state.
// Each bit in mask is set if the corresponding pointer-sized word of
// the base object is known to be zero.
// In other words, if mask & (1<<i) != 0, then [base+i*ptrSize, base+(i+1)*ptrSize)
// is known to be zero.
type ZeroRegion struct {
	Base *Value
	Mask uint64
}

// IsStackAddr reports whether v is known to be an address of a stack slot.
func IsStackAddr(v *Value) bool {
	for v.Op == ssaop.OpOffPtr || v.Op == ssaop.OpAddPtr || v.Op == ssaop.OpPtrIndex || v.Op == ssaop.OpCopy {
		v = v.Args[0]
	}
	switch v.Op {
	case ssaop.OpSP, ssaop.OpLocalAddr, ssaop.OpSelectNAddr, ssaop.OpGetCallerSP:
		return true
	}
	return false
}

// IsSanitizerSafeAddr reports whether v is known to be an address
// that doesn't need instrumentation.
func IsSanitizerSafeAddr(v *Value) bool {
	for v.Op == ssaop.OpOffPtr || v.Op == ssaop.OpAddPtr || v.Op == ssaop.OpPtrIndex || v.Op == ssaop.OpCopy {
		v = v.Args[0]
	}
	switch v.Op {
	case ssaop.OpSP, ssaop.OpLocalAddr, ssaop.OpSelectNAddr:
		// Stack addresses are always safe.
		return true
	case ssaop.OpITab, ssaop.OpStringPtr, ssaop.OpGetClosurePtr:
		// Itabs, string data, and closure fields are
		// read-only once initialized.
		return true
	case ssaop.OpAddr:
		vt := v.Aux.(*obj.LSym).Type
		return vt == objabi.SRODATA || vt == objabi.SLIBFUZZER_8BIT_COUNTER || vt == objabi.SCOVERAGE_COUNTER || vt == objabi.SCOVERAGE_AUXVAR
	}
	return false
}

// ComputeZeroMap returns a map from an ID of a memory value to
// a set of locations that are known to be zeroed at that memory value.
func (f *Func) ComputeZeroMap(select1 []*Value) map[ID]ZeroRegion {

	ptrSize := f.Config.PtrSize
	// Keep track of which parts of memory are known to be zero.
	// This helps with removing write barriers for various initialization patterns.
	// This analysis is conservative. We only keep track, for each memory state, of
	// which of the first 64 words of a single object are known to be zero.
	zeroes := map[ID]ZeroRegion{}
	// Find new objects.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if mem, ok := IsNewObject(v, select1); ok {
				// While compiling package runtime itself, we might see user
				// calls to newobject, which will have result type
				// unsafe.Pointer instead. We can't easily infer how large the
				// allocated memory is, so just skip it.
				if types.LocalPkg.Path == "runtime" && v.Type.IsUnsafePtr() {
					continue
				}

				nptr := min(64, v.Type.Elem().Size()/ptrSize)
				zeroes[mem.ID] = ZeroRegion{Base: v, Mask: 1<<uint(nptr) - 1}
			}
		}
	}
	// Find stores to those new objects.
	for {
		changed := false
		for _, b := range f.Blocks {
			// Note: iterating forwards helps convergence, as values are
			// typically (but not always!) in store order.
			for _, v := range b.Values {
				if v.Op != ssaop.OpStore {
					continue
				}
				z, ok := zeroes[v.MemoryArg().ID]
				if !ok {
					continue
				}
				ptr := v.Args[0]
				var off int64
				size := v.Aux.(*types.Type).Size()
				for ptr.Op == ssaop.OpOffPtr {
					off += ptr.AuxInt
					ptr = ptr.Args[0]
				}
				if ptr != z.Base {
					// Different base object - we don't know anything.
					// We could even be writing to the base object we know
					// about, but through an aliased but offset pointer.
					// So we have to throw all the zero information we have away.
					continue
				}
				// Round to cover any partially written pointer slots.
				// Pointer writes should never be unaligned like this, but non-pointer
				// writes to pointer-containing types will do this.
				if d := off % ptrSize; d != 0 {
					off -= d
					size += d
				}
				if d := size % ptrSize; d != 0 {
					size += ptrSize - d
				}
				// Clip to the 64 words that we track.
				minimum := max(off, 0)
				maximum := min(off+size, 64*ptrSize)

				// Clear bits for parts that we are writing (and hence
				// will no longer necessarily be zero).
				for i := minimum; i < maximum; i += ptrSize {
					bit := i / ptrSize
					z.Mask &^= 1 << uint(bit)
				}
				if z.Mask == 0 {
					// No more known zeros - don't bother keeping.
					continue
				}
				// Save updated known zero contents for new store.
				if zeroes[v.ID] != z {
					zeroes[v.ID] = z
					changed = true
				}
			}
		}
		if !changed {
			break
		}
	}
	if f.Pass.Debug > 0 {
		fmt.Printf("func %s\n", f.Name)
		for mem, z := range zeroes {
			fmt.Printf("  memory=v%d ptr=%v zeromask=%b\n", mem, z.Base, z.Mask)
		}
	}
	return zeroes
}
