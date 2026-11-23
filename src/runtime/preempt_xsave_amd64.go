// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/cpu"
	"internal/goexperiment"
	"unsafe"
)

// On amd64, asyncPreempt saves the vector and mask registers with XSAVEC if the
// CPU supports it. XSAVEC saves only the state components that aren't in their
// initial state, and records which ones it saved in the XSAVE header. Unlike
// XGETBV(ECX=1), XSAVEC is an SSE/AVX instruction, so it sees the thread's
// state even on OSes that load vector state lazily. asyncPreempt restores the
// registers with XRSTOR, which returns the components that weren't saved to
// their initial state.
//
// Without XSAVEC, asyncPreempt saves the registers that Go code may be using
// with explicit moves, in an xRegsNoXSAVEC.

// XSAVE state components that asyncPreempt saves. Component i corresponds to
// bit i of XCR0 and of XSTATE_BV (see the xcr0_AX_* constants). Components 0
// (x87) and 1 live in the legacy region, and the rest are in the extended
// region.
const (
	xsaveSSE       = 1 // X0-X15, in the legacy region
	xsaveYMM_Hi128 = 2 // Upper 128 bits of Y0-Y15
	xsaveOpMask    = 5 // K0-K7
	xsaveZMM_Hi256 = 6 // Upper 256 bits of Z0-Z15
	xsaveHi16_ZMM  = 7 // Z16-Z31

	xsaveFirstExtended = xsaveYMM_Hi128
	xsaveNumComponents = xsaveHi16_ZMM + 1
)

// Architectural offsets in an XSAVE area.
const (
	xsaveXMM       = 160 // X0-X15, in the legacy region. We ignore bytes 0–159 entirely.
	xsaveXSTATE_BV = 512 // XSTATE_BV, in the XSAVE header
	xsaveExtended  = 576 // Start of the extended region

	// xsaveMaxSize is the size of a compacted XSAVE area holding all of the
	// state components that asyncPreempt saves, given their architectural
	// sizes. Each of these sizes is a multiple of 64 bytes, so aligning
	// components adds no padding.
	xsaveMaxSize = (xsaveExtended +
		16*16 + // YMM_Hi128: upper 128 bits of Y0-Y15
		8*8 + // Opmask: K0-K7
		16*32 + // ZMM_Hi256: upper 256 bits of Z0-Z15
		16*64) // Hi16_ZMM: Z16-Z31
)

var (
	// useXSAVEC indicates that asyncPreempt saves registers with XSAVEC.
	useXSAVEC bool

	// xsaveMask is the set of XSAVE state components that asyncPreempt
	// saves with XSAVEC, as xcr0_AX_* bits.
	xsaveMask uint32

	// xsaveOff[i] and xsaveSize[i] are the offset and size of extended
	// XSAVE state component i in a compacted XSAVE area holding xsaveMask.
	xsaveOff, xsaveSize [xsaveNumComponents]uint16
)

// xRegInitCPU sets up the XSAVE area layout once CPU features are known.
func xRegInitCPU() {
	if !goexperiment.SIMDStable {
		// Without SIMD enabled, Go code only uses X0-X15. Use the fixed
		// xRegsNoXSAVEC layout, which is simpler.
		return
	}
	if !cpu.X86.HasXSAVEC {
		// Fall back to the fixed xRegsNoXSAVEC layout.
		return
	}
	if GOOS == "darwin" && cpu.X86.HasAVX512 {
		// darwin enables AVX-512 state per thread, on first use (see
		// go.dev/issue/49233). An XSAVE area saved on a thread with
		// AVX-512 enabled includes the AVX-512 components, and XRSTOR of
		// it on a thread without AVX-512 enabled faults with #GP, rather
		// than the #UD that darwin uses to enable AVX-512. Fall back to
		// the fixed xRegsNoXSAVEC layout, which uses EVEX instructions.
		return
	}
	// The vector state components that Go code may use.
	mask := uint32(xcr0_AX_SSE)
	if cpu.X86.HasAVX {
		mask |= xcr0_AX_YMM
	}
	if cpu.X86.HasAVX512 {
		mask |= xcr0_AX_OpMask | xcr0_AX_ZMM_Hi256 | xcr0_AX_Hi16_ZMM
	}
	// The compacted format lays out every requested extended component in
	// order, whether or not it's in use, aligning some of them. So the
	// offset of each component depends only on the requested components
	// and on their sizes and alignment, and is the same for every XSAVE
	// area in this process. XSAVEC doesn't write the components that
	// aren't in use, leaving holes in the area.
	off := uint32(xsaveExtended)
	for i := uint32(xsaveFirstExtended); i < xsaveNumComponents; i++ {
		if mask&(1<<i) == 0 {
			continue
		}
		size, align64 := cpu.XSAVEComponent(i)
		if align64 {
			off = (off + 63) &^ 63
		}
		xsaveOff[i], xsaveSize[i] = uint16(off), uint16(size)
		off += size
	}
	if off > xsaveMaxSize {
		// Unexpected component sizes. Fall back to the fixed xRegsNoXSAVEC
		// layout.
		return
	}
	xsaveMask = mask
	useXSAVEC = true
}

// base returns the address of x's XSAVE area.
func (x *xRegs) base() uintptr {
	return alignUp(uintptr(unsafe.Pointer(&x.buf)), 64)
}

// inUse returns the set of state components saved in x's XSAVE area, as
// xcr0_AX_* bits.
func (x *xRegs) inUse() uint32 {
	return *(*uint32)(unsafe.Pointer(x.base() + xsaveXSTATE_BV))
}

func (dst *xRegs) copyFrom(src *xRegs) {
	if !useXSAVEC {
		memmove(unsafe.Pointer(&dst.buf), unsafe.Pointer(&src.buf), unsafe.Sizeof(xRegsNoXSAVEC{}))
		return
	}
	// Copy the legacy region and the XSAVE header, and each extended
	// component that's in use. XRSTOR ignores the other components.
	d, s := dst.base(), src.base()
	memmove(unsafe.Pointer(d), unsafe.Pointer(s), xsaveExtended)
	inUse := src.inUse()
	for i := xsaveFirstExtended; i < xsaveNumComponents; i++ {
		if inUse&(1<<i) != 0 {
			off := uintptr(xsaveOff[i])
			memmove(unsafe.Pointer(d+off), unsafe.Pointer(s+off), uintptr(xsaveSize[i]))
		}
	}
}

func (x *xRegs) scanRanges(yield func(base, size uintptr)) {
	if !useXSAVEC {
		yield(uintptr(unsafe.Pointer(&x.buf)), unsafe.Sizeof(xRegsNoXSAVEC{}))
		return
	}
	// Scan the vector registers that are in use. K0-K7 can't hold pointers.
	b, inUse := x.base(), x.inUse()
	if inUse&(1<<xsaveSSE) != 0 {
		yield(b+xsaveXMM, 16*16)
	}
	for _, i := range [...]int{xsaveYMM_Hi128, xsaveZMM_Hi256, xsaveHi16_ZMM} {
		if inUse&(1<<i) != 0 {
			yield(b+uintptr(xsaveOff[i]), uintptr(xsaveSize[i]))
		}
	}
}
