// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit

package jit

import (
	"encoding/binary"
	"runtime"
)

// instructionBlock makes each exported emission atomic with respect to buffer
// exhaustion. Its bytes and all metadata are committed only after construction.
// No callbacks, allocation or interfaces are used by the generated code itself.
type instructionBlock struct{ bytes []byte }

func (b *instructionBlock) byte(v ...byte) { b.bytes = append(b.bytes, v...) }
func (b *instructionBlock) u32(v uint32)   { b.bytes = binary.LittleEndian.AppendUint32(b.bytes, v) }
func (b *instructionBlock) u64(v uint64)   { b.bytes = binary.LittleEndian.AppendUint64(b.bytes, v) }

func (b *instructionBlock) move(dst, src PhysicalRegister) {
	if dst.Class != IntegerRegister || src.Class != IntegerRegister {
		panic("runtime/jit: expected integer registers")
	}
	d, s := uint32(dst.Number), uint32(src.Number)
	switch runtime.GOARCH {
	case "amd64":
		b.byte(0x48|byte(s>>3)<<2|byte(d>>3), 0x89, 0xc0|byte(s&7)<<3|byte(d&7))
	case "arm64":
		if d == 31 || s == 31 {
			b.u32(0x91000000 | s<<5 | d)
		} else {
			b.u32(0xaa0003e0 | s<<16 | d)
		}
	case "riscv64":
		b.u32(s<<15 | d<<7 | 0x13)
	}
}
func (b *instructionBlock) immediate(dst PhysicalRegister, v uint64) {
	d := uint32(dst.Number)
	switch runtime.GOARCH {
	case "amd64":
		b.byte(0x48|byte(d>>3), 0xb8|byte(d&7))
		b.u64(v)
	case "arm64":
		b.u32(0xd2800000 | uint32(v&65535)<<5 | d)
		for i := uint32(1); i < 4; i++ {
			b.u32(0xf2800000 | i<<21 | uint32(v>>(16*i)&65535)<<5 | d)
		}
	case "riscv64":
		// Fixed-length radix-2048 materialization avoids embedded literals and
		// unaligned LDs when the instruction starts four bytes past an 8-byte boundary.
		b.u32(uint32(v>>55)<<20 | d<<7 | 0x13)
		for shift := 44; shift >= 0; shift -= 11 {
			b.u32(11<<20 | d<<15 | 1<<12 | d<<7 | 0x13)
			b.u32(uint32(v>>uint(shift)&2047)<<20 | d<<15 | 6<<12 | d<<7 | 0x13)
		}
	}
}
func (b *instructionBlock) memory(load bool, reg, base PhysicalRegister, off int32) {
	r, a := uint32(reg.Number), uint32(base.Number)
	if reg.Class == FloatRegister {
		switch runtime.GOARCH {
		case "amd64":
			if load {
				b.byte(0xf3)
			} else {
				b.byte(0x66)
			}
			b.byte(0x40|byte(r>>3)<<2|byte(a>>3), 0x0f)
			if load {
				b.byte(0x7e)
			} else {
				b.byte(0xd6)
			}
			b.byte(0x80 | byte(r&7)<<3 | byte(a&7))
			if a&7 == 4 {
				b.byte(0x24)
			}
			b.u32(uint32(off))
		case "arm64":
			if off < 0 || off%8 != 0 || off/8 >= 4096 {
				panic("runtime/jit: FP spill out of range")
			}
			op := uint32(0xfd000000)
			if load {
				op = 0xfd400000
			}
			b.u32(op | uint32(off/8)<<10 | a<<5 | r)
		case "riscv64":
			if off < 0 || off > 2047 {
				panic("runtime/jit: FP spill out of range")
			}
			v := uint32(off)
			if load {
				b.u32(v<<20 | a<<15 | 3<<12 | r<<7 | 7)
			} else {
				b.u32(v>>5<<25 | r<<20 | a<<15 | 3<<12 | (v&31)<<7 | 0x27)
			}
		}
		return
	}
	switch runtime.GOARCH {
	case "amd64":
		op := byte(0x89)
		if load {
			op = 0x8b
		}
		b.byte(0x48|byte(r>>3)<<2|byte(a>>3), op, 0x80|byte(r&7)<<3|byte(a&7))
		if a&7 == 4 {
			b.byte(0x24)
		}
		b.u32(uint32(off))
	case "arm64":
		if off >= 0 && off%8 == 0 && off/8 < 4096 {
			op := uint32(0xf9000000)
			if load {
				op = 0xf9400000
			}
			b.u32(op | uint32(off/8)<<10 | a<<5 | r)
			return
		}
		if off < -256 || off > 255 {
			panic("runtime/jit: frame offset out of range")
		}
		op := uint32(0xf8000000)
		if load {
			op = 0xf8400000
		}
		b.u32(op | uint32(off&511)<<12 | a<<5 | r)
	case "riscv64":
		if off < -2048 || off > 2047 {
			panic("runtime/jit: frame offset out of range")
		}
		if load {
			b.u32(uint32(off&4095)<<20 | a<<15 | 3<<12 | r<<7 | 3)
		} else {
			v := uint32(off) & 4095
			b.u32(v>>5<<25 | r<<20 | a<<15 | 3<<12 | v&31<<7 | 0x23)
		}
	}
}
func (b *instructionBlock) stackAdjust(delta int32, scratch PhysicalRegister) {
	r := nativeRegisterConfig()
	sp := r.Stack

	switch runtime.GOARCH {
	case "amd64":
		op := byte(0xc4)
		v := delta
		if v <= 0 {
			op = 0xec
			v = -v
		}
		b.byte(0x48, 0x81, op)
		b.u32(uint32(v))
	case "arm64":
		v := int64(delta)
		if v < 0 {
			v = -v
		}
		b.immediate(scratch, uint64(v))
		op := uint32(0x8b206000)
		if delta <= 0 {
			op = 0xcb206000
		}
		b.u32(op | uint32(scratch.Number)<<16 | 31<<5 | 31)
	case "riscv64":
		b.immediate(scratch, uint64(int64(delta)))
		b.u32(uint32(scratch.Number)<<20 | uint32(sp.Number)<<15 | uint32(sp.Number)<<7 | 0x33)
	}
}
func (b *instructionBlock) call(target PhysicalRegister) {
	n := uint32(target.Number)
	switch runtime.GOARCH {
	case "amd64":
		b.byte(0x40|byte(n>>3), 0xff, 0xd0|byte(n&7))
	case "arm64":
		b.u32(0xd63f0000 | n<<5)
	case "riscv64":
		b.u32(n<<15 | 1<<7 | 0x67)
	}
}
func (b *instructionBlock) ret() {
	switch runtime.GOARCH {
	case "amd64":
		b.byte(0xc3)
	case "arm64":
		b.u32(0xd65f03c0)
	case "riscv64":
		b.u32(1<<15 | 0x67)
	}
}

// FramePlan defines an ordinary JIT frame. LocalBytes must be a multiple of the
// target stack alignment and include every local/spill slot. Frame may override
// the default frame base (for example when it overlaps ABI argument registers).
type FramePlan struct {
	LocalBytes int32
	Frame      PhysicalRegister
}

// FrameFixup is an opaque patch handle for a one-pass function whose final local
// frame size is known only after its body has been emitted.
type FrameFixup struct {
	offset, length int
	frame          PhysicalRegister
}

func frameRegister(frame PhysicalRegister) PhysicalRegister {
	if frame == (PhysicalRegister{}) {
		frame = nativeRegisterConfig().Frame
	}
	r := nativeRegisterConfig()
	if frame.Class != IntegerRegister || frame == r.Stack || frame == r.Goroutine || frame == r.Closure || frame.Number >= gpRegisterLimit() {
		panic("runtime/jit: invalid frame register")
	}
	return frame
}

func frameCode(p FramePlan) instructionBlock {
	if p.LocalBytes < 0 || p.LocalBytes > 1<<30 || p.LocalBytes%16 != 0 {
		panic("runtime/jit: unaligned frame size")
	}
	f := frameRegister(p.Frame)
	r := nativeRegisterConfig()
	var b instructionBlock
	switch runtime.GOARCH {
	case "amd64":
		if f.Number >= 8 {
			b.byte(0x41)
		}
		b.byte(0x50 | f.Number&7)
		b.move(f, r.Stack)
	case "arm64":
		b.u32(0xa9bf0000 | 30<<10 | 31<<5 | uint32(f.Number))
		b.move(f, r.Stack)
	case "riscv64":
		b.u32(uint32((-16)&4095)<<20 | 2<<15 | 2<<7 | 0x13)
		b.memory(false, f, r.Stack, 0)
		b.memory(false, gp(1), r.Stack, 8)
		b.move(f, r.Stack)
	}
	// Always include an adjustment, even for zero, to keep fixups fixed-size.
	b.stackAdjust(-p.LocalBytes, r.Scratch)
	return b
}

// EmitFrame emits a prologue and returns a handle for PatchFrame. It preserves
// argument and closure registers. Its scratch register is clobbered.
func EmitFrame(dst CodeBuffer, p FramePlan) FrameFixup {
	b := frameCode(p)
	f := FrameFixup{dst.Offset(), len(b.bytes), frameRegister(p.Frame)}
	dst.Write(b.bytes)
	return f
}

// PatchFrame replaces only the prologue reserved by EmitFrame.
func PatchFrame(dst CodeBuffer, f FrameFixup, localBytes int32) {
	b := frameCode(FramePlan{localBytes, f.frame})
	if len(b.bytes) != f.length {
		panic("runtime/jit: frame patch length changed")
	}
	dst.Patch(f.offset, b.bytes)
}

// EmitReturn restores a frame and returns with ABI result registers intact.
func EmitReturn(dst CodeBuffer, frame PhysicalRegister) {
	var b instructionBlock
	b.leave(frameRegister(frame))
	b.ret()
	dst.Write(b.bytes)
}

// EmitLeafReturn emits a return for a function which has not established a frame
// and has not changed SP or its link register.
func EmitLeafReturn(dst CodeBuffer) {
	nativeRegisterConfig()
	var b instructionBlock
	b.ret()
	dst.Write(b.bytes)
}

// CallPlan describes a prepared call boundary. The client has already placed
// arguments and saved every live value in CallClobbers. This helper emits the
// CALL only; EmitCallFrame can first reserve the caller-owned ABI area.
// Exactly one of Entry, Target or Closure must select the target. PhysicalRegister zero
// is valid: use Indirect to select Target. Closure points at a Go function value's
// context allocation, not at a variable containing the function value.
type CallPlan struct {
	Entry    uintptr
	Target   PhysicalRegister
	Indirect bool
	Closure  *PhysicalRegister
}

// CallEmission identifies the exact return PC and overwritten registers. The
// client must combine it with local liveness before publishing a stack map.
type CallEmission struct {
	Start, ReturnPC, End int
	Clobbers             RegisterSet
}

// EmitCall emits a direct, register-indirect, or closure call atomically.
func EmitCall(dst CodeBuffer, p CallPlan) CallEmission {
	r := nativeRegisterConfig()
	var b instructionBlock
	target := p.Target
	if p.Closure != nil {
		if p.Entry != 0 || p.Indirect {
			panic("runtime/jit: multiple call targets")
		}
		b.move(r.Closure, *p.Closure)
		b.memory(true, r.Scratch, r.Closure, int32(FunctionEntryOffset()))
		target = r.Scratch
	} else if !p.Indirect {
		if p.Entry == 0 {
			panic("runtime/jit: zero call target")
		}
		target = r.Scratch
		b.immediate(target, uint64(p.Entry))
	}
	if target.Class != IntegerRegister || target.Number >= gpRegisterLimit() {
		panic("runtime/jit: invalid call target register")
	}
	b.call(target)
	start := dst.Offset()
	dst.Write(b.bytes)
	return CallEmission{start, dst.Offset(), dst.Offset(), r.CallClobbers}
}

// EmitCallFrame reserves the ABI argument/result/spill area. Scratch is clobbered;
// it must not hold a prepared argument. The returned size is passed to ReleaseCallFrame.
func EmitCallFrame(dst CodeBuffer, c CallingConvention) int32 {
	if c.CallFrameSize > 1<<30 {
		panic("runtime/jit: call frame too large")
	}
	var b instructionBlock
	b.stackAdjust(-int32(c.CallFrameSize), nativeRegisterConfig().Scratch)
	dst.Write(b.bytes)
	return int32(c.CallFrameSize)
}

// ReleaseCallFrame restores SP after results have been consumed. Scratch is clobbered.
func ReleaseCallFrame(dst CodeBuffer, size int32) {
	if size < 0 || size%16 != 0 {
		panic("runtime/jit: invalid call frame release")
	}
	var b instructionBlock
	b.stackAdjust(size, nativeRegisterConfig().Scratch)
	dst.Write(b.bytes)
}

// FrameStackMap builds the unwind recipe for a frame emitted by EmitFrame.
// frameBytes is the complete SP-to-frame-base distance at this safepoint,
// including transient outgoing call areas. pointerMask describes this range
// and the saved frame pointer word; the latter is relocated separately too.
func FrameStackMap(returnPC uintptr, frameBytes uintptr, pointerMask []byte) StackMap {
	nativeRegisterConfig()
	if frameBytes%8 != 0 {
		panic("runtime/jit: unaligned safepoint frame")
	}
	return StackMap{PCOffset: returnPC, FrameWords: frameBytes/8 + 1, PointerMask: pointerMask, HasUnwind: true, UnwindBaseOffset: frameBytes, CallerPCOffset: 8, CallerSPOffset: 16}
}

func (b *instructionBlock) jump(base, target int) {
	pos := base + len(b.bytes)
	switch runtime.GOARCH {
	case "amd64":
		b.byte(0xe9)
		b.u32(uint32(int32(target - pos - 5)))
	case "arm64":
		d := target - pos
		if d%4 != 0 || d < -(1<<27) || d >= 1<<27 {
			panic("runtime/jit: jump out of range")
		}
		b.u32(0x14000000 | uint32(d/4)&0x3ffffff)
	case "riscv64":
		d := int32(target - pos)
		if d%2 != 0 || d < -(1<<20) || d >= 1<<20 {
			panic("runtime/jit: jump out of range")
		}
		v := uint32(d)
		b.u32((v>>20&1)<<31 | (v>>1&1023)<<21 | (v>>11&1)<<20 | (v>>12&255)<<12 | 0x6f)
	}
}

// StackCheckFixup holds a fixed-size stack check for one-pass frame sizing.
type StackCheckFixup struct{ offset, length int }

func stackCheckCode(base int, frameBytes uintptr, grow int) instructionBlock {
	if frameBytes > 1<<30 {
		panic("runtime/jit: stack check too large")
	}
	r := nativeRegisterConfig()
	config := StackCheckConfig()
	var b instructionBlock
	demand := uintptr(0)
	if frameBytes > config.StackSmall {
		demand = frameBytes - config.StackSmall
	}
	switch runtime.GOARCH {
	case "amd64":
		b.move(r.Scratch, r.Stack)
		b.byte(0x49, 0x81, 0xec)
		b.u32(uint32(demand)) // SUB R12,demand
		pos := base + len(b.bytes)
		b.byte(0x0f, 0x82)
		b.u32(uint32(int32(grow - pos - 6)))
		b.byte(0x4d, 0x3b, 0xa6)
		b.u32(uint32(config.StackGuardOffset)) // CMP R12,[R14+guard]
		pos = base + len(b.bytes)
		b.byte(0x0f, 0x86)
		b.u32(uint32(int32(grow - pos - 6)))
	case "arm64":
		b.immediate(r.Scratch, uint64(demand))
		b.move(gp(16), r.Stack)
		b.u32(0xeb00001f | uint32(r.Scratch.Number)<<16 | 16<<5) // CMP SP-copy,demand
		b.u32(0x54000042)                                        // B.HS +8
		b.jump(base, grow)
		b.u32(0xcb000000 | uint32(r.Scratch.Number)<<16 | 16<<5 | 16)
		b.memory(true, r.Scratch, r.Goroutine, int32(config.StackGuardOffset))
		b.u32(0xeb00001f | uint32(r.Scratch.Number)<<16 | 16<<5)
		b.u32(0x54000048) // B.HI +8
		b.jump(base, grow)
	case "riscv64":
		b.immediate(r.Scratch, uint64(demand))
		b.u32(31<<20 | 2<<15 | 7<<12 | 4<<8 | 0x63) // BGEU SP,demand,+8
		b.jump(base, grow)
		b.u32(0x40000033 | 31<<20 | 2<<15 | 30<<7) // SUB X30,SP,demand
		b.memory(true, r.Scratch, r.Goroutine, int32(config.StackGuardOffset))
		b.u32(30<<20 | 31<<15 | 6<<12 | 4<<8 | 0x63) // BLTU guard,X30,+8
		b.jump(base, grow)
	}
	return b
}

// EmitStackCheck checks the complete worst-case frame demand, including frame
// records and outgoing call areas, before establishing a frame. growOffset and
// retry offsets are CodeBuffer-relative byte offsets. Scratch is clobbered;
// arm64 also clobbers X16 and riscv64 X30. No argument registers are changed.
func EmitStackCheck(dst CodeBuffer, frameBytes uintptr, growOffset int) StackCheckFixup {
	b := stackCheckCode(dst.Offset(), frameBytes, growOffset)
	f := StackCheckFixup{dst.Offset(), len(b.bytes)}
	dst.Write(b.bytes)
	return f
}

// PatchStackCheck resolves the final demand and slow-path target.
func PatchStackCheck(dst CodeBuffer, f StackCheckFixup, frameBytes uintptr, growOffset int) {
	b := stackCheckCode(f.offset, frameBytes, growOffset)
	if len(b.bytes) != f.length {
		panic("runtime/jit: stack check patch length changed")
	}
	dst.Patch(f.offset, b.bytes)
}

func (b *instructionBlock) leave(frame PhysicalRegister) {
	if runtime.GOARCH == "amd64" && frame.Number == 5 {
		b.byte(0xc9)
		return
	}
	r := nativeRegisterConfig()
	b.move(r.Stack, frame)
	switch runtime.GOARCH {
	case "amd64":
		if frame.Number >= 8 {
			b.byte(0x41)
		}
		b.byte(0x58 | frame.Number&7)
	case "arm64":
		b.u32(0xa8c10000 | 30<<10 | 31<<5 | uint32(frame.Number))
	case "riscv64":
		b.memory(true, frame, r.Stack, 0)
		b.memory(true, gp(1), r.Stack, 8)
		b.u32(16<<20 | 2<<15 | 2<<7 | 0x13)
	}
}

// EmitStackGrow spills all incoming ABI registers into a precisely described
// temporary frame, calls morestack using its special platform convention,
// restores the arguments, and jumps to retryOffset. The returned map must be
// published with the function before execution. Context is preserved by
// morestack. Frame must not overlap an incoming argument register.
func EmitStackGrow(dst CodeBuffer, c CallingConvention, frame PhysicalRegister, retryOffset int) StackMap {
	f := frameRegister(frame)
	r := nativeRegisterConfig()
	var parts []ValuePart
	for _, v := range c.Inputs {
		for _, p := range v.Parts {
			if p.Location.Kind == InRegister {
				if p.Location.Register == f {
					panic("runtime/jit: growth frame overlaps argument")
				}
				parts = append(parts, p)
			}
		}
	}
	size := int32(alignABI(uintptr(len(parts))*8, 16))
	b := frameCode(FramePlan{size, f})
	for i, p := range parts {
		b.memory(false, p.Location.Register, r.Stack, int32(i*8))
	}
	b.immediate(r.Scratch, uint64(StackCheckConfig().MoreStackPC))
	if runtime.GOARCH == "riscv64" {
		b.memory(true, gp(1), f, 8)
		b.u32(uint32(r.Scratch.Number)<<15 | 5<<7 | 0x67)
	} else if runtime.GOARCH == "arm64" {
		// morestack uses R3 to recover the interrupted function's caller LR.
		b.memory(true, gp(3), f, 8)
		b.call(r.Scratch)
	} else {
		b.call(r.Scratch)
	}
	returnPC := dst.Offset() + len(b.bytes)
	for i, p := range parts {
		b.memory(true, p.Location.Register, r.Stack, int32(i*8))
	}
	b.leave(f)
	b.jump(dst.Offset(), retryOffset)
	mask := make([]byte, (int(size)/8+1+7)/8)
	for i, p := range parts {
		if p.Pointer {
			mask[i/8] |= 1 << uint(i%8)
		}
	}
	m := FrameStackMap(uintptr(returnPC), uintptr(size), mask)
	dst.Write(b.bytes)
	return m
}

// Yield is an actual cooperative scheduling boundary, unlike Preempt's polling
// query. Call it through an ordinary described Go-call safepoint after spilling
// live pointers. It is safe to call even if the preemption request was cleared.
func Yield() { runtime.Gosched() }

func gpRegisterLimit() uint8 {
	if runtime.GOARCH == "amd64" {
		return 16
	}
	return 32
}

// EmitLeave restores the caller frame without returning. It is useful for
// standalone fragments whose caller appends its own final return instruction.
func EmitLeave(dst CodeBuffer, frame PhysicalRegister) {
	var b instructionBlock
	b.leave(frameRegister(frame))
	dst.Write(b.bytes)
}
