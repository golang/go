// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect

import (
	"internal/abi"
	"internal/goarch"
	"unsafe"
)

// These variables are used by the register assignment
// algorithm in this file.
//
// They should be modified with care (no other reflect code
// may be executing) and are generally only modified
// when testing this package.
//
// They should never be set higher than their internal/abi
// constant counterparts, because the system relies on a
// structure that is at least large enough to hold the
// registers the system supports.
//
// Currently they're set to zero because using the actual
// constants will break every part of the toolchain that
// uses reflect to call functions (e.g. go test, or anything
// that uses text/template). The values that are currently
// commented out there should be the actual values once
// we're ready to use the register ABI everywhere.
var (
	intArgRegs   = abi.IntArgRegs
	floatArgRegs = abi.FloatArgRegs
	floatRegSize = uintptr(abi.EffectiveFloatRegSize)
)

// abiStep represents an ABI "instruction." Each instruction
// describes one part of how to translate between a Go value
// in memory and a call frame.
type abiStep struct {
	kind abiStepKind

	// offset and size together describe a part of a Go value
	// in memory.
	offset uintptr
	size   uintptr // size in bytes of the part

	// These fields describe the ABI side of the translation.
	stkOff uintptr // stack offset, used if kind == abiStepStack
	ireg   int     // integer register index, used if kind == abiStepIntReg or kind == abiStepPointer
	freg   int     // FP register index, used if kind == abiStepFloatReg
}

// abiStepKind is the "op-code" for an abiStep instruction.
type abiStepKind int

const (
	abiStepBad      abiStepKind = iota
	abiStepStack                // copy to/from stack
	abiStepIntReg               // copy to/from integer register
	abiStepPointer              // copy pointer to/from integer register
	abiStepFloatReg             // copy to/from FP register
)

// abiSeq represents a sequence of ABI instructions for copying
// from a series of reflect.Values to a call frame (for call arguments)
// or vice-versa (for call results).
//
// An abiSeq should be populated by calling its addArg method.
type abiSeq struct {
	// steps is the set of instructions.
	//
	// The instructions are grouped together by whole arguments,
	// with the starting index for the instructions
	// of the i'th Go value available in valueStart.
	//
	// For instance, if this abiSeq represents 3 arguments
	// passed to a function, then the 2nd argument's steps
	// begin at steps[valueStart[1]].
	//
	// Because reflect accepts Go arguments in distinct
	// Values and each Value is stored separately, each abiStep
	// that begins a new argument will have its offset
	// field == 0.
	steps      []abiStep
	valueStart []int

	stackBytes   uintptr // stack space used
	iregs, fregs int     // registers used
}

func (a *abiSeq) dump() {
	for i, p := range a.steps {
		println("part", i, p.kind, p.offset, p.size, p.stkOff, p.ireg, p.freg)
	}
	print("values ")
	for _, i := range a.valueStart {
		print(i, " ")
	}
	println()
	println("stack", a.stackBytes)
	println("iregs", a.iregs)
	println("fregs", a.fregs)
}

// stepsForValue returns the ABI instructions for translating
// the i'th Go argument or return value represented by this
// abiSeq to the Go ABI.
func (a *abiSeq) stepsForValue(i int) []abiStep {
	s := a.valueStart[i]
	var e int
	if i == len(a.valueStart)-1 {
		e = len(a.steps)
	} else {
		e = a.valueStart[i+1]
	}
	return a.steps[s:e]
}

// addArg extends the abiSeq with a new Go value of type t.
//
// If the value was stack-assigned, returns the single
// abiStep describing that translation, and nil otherwise.
func (a *abiSeq) addArg(t *rtype) *abiStep {
	// We'll always be adding a new value, so do that first.
	pStart := len(a.steps)
	a.valueStart = append(a.valueStart, pStart)
	if t.size == 0 {
		// If the size of the argument type is zero, then
		// in order to degrade gracefully into ABI0, we need
		// to stack-assign this type. The reason is that
		// although zero-sized types take up no space on the
		// stack, they do cause the next argument to be aligned.
		// So just do that here, but don't bother actually
		// generating a new ABI step for it (there's nothing to
		// actually copy).
		//
		// We cannot handle this in the recursive case of
		// regAssign because zero-sized *fields* of a
		// non-zero-sized struct do not cause it to be
		// stack-assigned. So we need a special case here
		// at the top.
		a.stackBytes = align(a.stackBytes, uintptr(t.align))
		return nil
	}
	// Hold a copy of "a" so that we can roll back if
	// register assignment fails.
	aOld := *a
	if !a.regAssign(t, 0) {
		// Register assignment failed. Roll back any changes
		// and stack-assign.
		*a = aOld
		a.stackAssign(t.size, uintptr(t.align))
		return &a.steps[len(a.steps)-1]
	}
	return nil
}

// addRcvr extends the abiSeq with a new method call
// receiver according to the interface calling convention.
//
// If the receiver was stack-assigned, returns the single
// abiStep describing that translation, and nil otherwise.
// Returns true if the receiver is a pointer.
func (a *abiSeq) addRcvr(rcvr *rtype) (*abiStep, bool) {
	// The receiver is always one word.
	a.valueStart = append(a.valueStart, len(a.steps))
	var ok, ptr bool
	if ifaceIndir(rcvr) || rcvr.pointers() {
		ok = a.assignIntN(0, goarch.PtrSize, 1, 0b1)
		ptr = true
	} else {
		// TODO(mknyszek): Is this case even possible?
		// The interface data work never contains a non-pointer
		// value. This case was copied over from older code
		// in the reflect package which only conditionally added
		// a pointer bit to the reflect.(Value).Call stack frame's
		// GC bitmap.
		ok = a.assignIntN(0, goarch.PtrSize, 1, 0b0)
		ptr = false
	}
	if !ok {
		a.stackAssign(goarch.PtrSize, goarch.PtrSize)
		return &a.steps[len(a.steps)-1], ptr
	}
	return nil, ptr
}

// regAssign attempts to reserve argument registers for a value of
// type t, stored at some offset.
//
// It returns whether or not the assignment succeeded, but
// leaves any changes it made to a.steps behind, so the caller
// must undo that work by adjusting a.steps if it fails.
//
// This method along with the assign* methods represent the
// complete register-assignment algorithm for the Go ABI.
func (a *abiSeq) regAssign(t *rtype, offset uintptr) bool {
	switch t.Kind() {
	case UnsafePointer, Pointer, Chan, Map, Func:
		return a.assignIntN(offset, t.size, 1, 0b1)
	case Bool, Int, Uint, Int8, Uint8, Int16, Uint16, Int32, Uint32, Uintptr:
		return a.assignIntN(offset, t.size, 1, 0b0)
	case Int64, Uint64:
		switch goarch.PtrSize {
		case 4:
			return a.assignIntN(offset, 4, 2, 0b0)
		case 8:
			return a.assignIntN(offset, 8, 1, 0b0)
		}
	case Float32, Float64:
		return a.assignFloatN(offset, t.size, 1)
	case Complex64:
		return a.assignFloatN(offset, 4, 2)
	case Complex128:
		return a.assignFloatN(offset, 8, 2)
	case String:
		return a.assignIntN(offset, goarch.PtrSize, 2, 0b01)
	case Interface:
		return a.assignIntN(offset, goarch.PtrSize, 2, 0b10)
	case Slice:
		return a.assignIntN(offset, goarch.PtrSize, 3, 0b001)
	case Array:
		tt := (*arrayType)(unsafe.Pointer(t))
		switch tt.len {
		case 0:
			// There's nothing to assign, so don't modify
			// a.steps but succeed so the caller doesn't
			// try to stack-assign this value.
			return true
		case 1:
			return a.regAssign(tt.elem, offset)
		default:
			return false
		}
	case Struct:
		st := (*structType)(unsafe.Pointer(t))
		for i := range st.fields {
			f := &st.fields[i]
			if !a.regAssign(f.typ, offset+f.offset()) {
				return false
			}
		}
		return true
	default:
		print("t.Kind == ", t.Kind(), "\n")
		panic("unknown type kind")
	}
	panic("unhandled register assignment path")
}

// assignIntN assigns n values to registers, each "size" bytes large,
// from the data at [offset, offset+n*size) in memory. Each value at
// [offset+i*size, offset+(i+1)*size) for i < n is assigned to the
// next n integer registers.
//
// Bit i in ptrMap indicates whether the i'th value is a pointer.
// n must be <= 8.
//
// Returns whether assignment succeeded.
func (a *abiSeq) assignIntN(offset, size uintptr, n int, ptrMap uint8) bool {
	if n > 8 || n < 0 {
		panic("invalid n")
	}
	if ptrMap != 0 && size != goarch.PtrSize {
		panic("non-empty pointer map passed for non-pointer-size values")
	}
	if a.iregs+n > intArgRegs {
		return false
	}
	for i := 0; i < n; i++ {
		kind := abiStepIntReg
		if ptrMap&(uint8(1)<<i) != 0 {
			kind = abiStepPointer
		}
		a.steps = append(a.steps, abiStep{
			kind:   kind,
			offset: offset + uintptr(i)*size,
			size:   size,
			ireg:   a.iregs,
		})
		a.iregs++
	}
	return true
}

// assignFloatN assigns n values to registers, each "size" bytes large,
// from the data at [offset, offset+n*size) in memory. Each value at
// [offset+i*size, offset+(i+1)*size) for i < n is assigned to the
// next n floating-point registers.
//
// Returns whether assignment succeeded.
func (a *abiSeq) assignFloatN(offset, size uintptr, n int) bool {
	if n < 0 {
		panic("invalid n")
	}
	if a.fregs+n > floatArgRegs || floatRegSize < size {
		return false
	}
	for i := 0; i < n; i++ {
		a.steps = append(a.steps, abiStep{
			kind:   abiStepFloatReg,
			offset: offset + uintptr(i)*size,
			size:   size,
			freg:   a.fregs,
		})
		a.fregs++
	}
	return true
}

// stackAssign reserves space for one value that is "size" bytes
// large with alignment "alignment" to the stack.
//
// Should not be called directly; use addArg instead.
func (a *abiSeq) stackAssign(size, alignment uintptr) {
	a.stackBytes = align(a.stackBytes, alignment)
	a.steps = append(a.steps, abiStep{
		kind:   abiStepStack,
		offset: 0, // Only used for whole arguments, so the memory offset is 0.
		size:   size,
		stkOff: a.stackBytes,
	})
	a.stackBytes += size
}

// abiDesc describes the ABI for a function or method.
type abiDesc struct {
	// call and ret represent the translation steps for
	// the call and return paths of a Go function.
	call, ret abiSeq

	// These fields describe the stack space allocated
	// for the call. stackCallArgsSize is the amount of space
	// reserved for arguments but not return values. retOffset
	// is the offset at which return values begin, and
	// spill is the size in bytes of additional space reserved
	// to spill argument registers into in case of preemption in
	// reflectcall's stack frame.
	stackCallArgsSize, retOffset, spill uintptr

	// stackPtrs is a bitmap that indicates whether
	// each word in the ABI stack space (stack-assigned
	// args + return values) is a pointer. Used
	// as the heap pointer bitmap for stack space
	// passed to reflectcall.
	stackPtrs *bitVector

	// inRegPtrs is a bitmap whose i'th bit indicates
	// whether the i'th integer argument register contains
	// a pointer. Used by makeFuncStub and methodValueCall
	// to make result pointers visible to the GC.
	//
	// outRegPtrs is the same, but for result values.
	// Used by reflectcall to make result pointers visible
	// to the GC.
	inRegPtrs, outRegPtrs abi.IntArgRegBitmap
}

func (a *abiDesc) dump() {
	println("ABI")
	println("call")
	a.call.dump()
	println("ret")
	a.ret.dump()
	println("stackCallArgsSize", a.stackCallArgsSize)
	println("retOffset", a.retOffset)
	println("spill", a.spill)
	print("inRegPtrs:")
	dumpPtrBitMap(a.inRegPtrs)
	println()
	print("outRegPtrs:")
	dumpPtrBitMap(a.outRegPtrs)
	println()
}

func dumpPtrBitMap(b abi.IntArgRegBitmap) {
	for i := 0; i < intArgRegs; i++ {
		x := 0
		if b.Get(i) {
			x = 1
		}
		print(" ", x)
	}
}

func newAbiDesc(t *funcType, rcvr *rtype) abiDesc {
	// We need to add space for this argument to
	// the frame so that it can spill args into it.
	//
	// The size of this space is just the sum of the sizes
	// of each register-allocated type.
	//
	// TODO(mknyszek): Remove this when we no longer have
	// caller reserved spill space.
	spill := uintptr(0)

	// Compute gc program & stack bitmap for stack arguments
	stackPtrs := new(bitVector)

	// Compute the stack frame pointer bitmap and register
	// pointer bitmap for arguments.
	inRegPtrs := abi.IntArgRegBitmap{}

	// Compute abiSeq for input parameters.
	var in abiSeq
	if rcvr != nil {
		stkStep, isPtr := in.addRcvr(rcvr)
		if stkStep != nil {
			if isPtr {
				stackPtrs.append(1)
			} else {
				stackPtrs.append(0)
			}
		} else {
			spill += goarch.PtrSize
		}
	}
	for i, arg := range t.in() {
		stkStep := in.addArg(arg)
		if stkStep != nil {
			addTypeBits(stackPtrs, stkStep.stkOff, arg)
		} else {
			spill = align(spill, uintptr(arg.align))
			spill += arg.size
			for _, st := range in.stepsForValue(i) {
				if st.kind == abiStepPointer {
					inRegPtrs.Set(st.ireg)
				}
			}
		}
	}
	spill = align(spill, goarch.PtrSize)

	// From the input parameters alone, we now know
	// the stackCallArgsSize and retOffset.
	stackCallArgsSize := in.stackBytes
	retOffset := align(in.stackBytes, goarch.PtrSize)

	// Compute the stack frame pointer bitmap and register
	// pointer bitmap for return values.
	outRegPtrs := abi.IntArgRegBitmap{}

	// Compute abiSeq for output parameters.
	var out abiSeq
	// Stack-assigned return values do not share
	// space with arguments like they do with registers,
	// so we need to inject a stack offset here.
	// Fake it by artificially extending stackBytes by
	// the return offset.
	out.stackBytes = retOffset
	for i, res := range t.out() {
		stkStep := out.addArg(res)
		if stkStep != nil {
			addTypeBits(stackPtrs, stkStep.stkOff, res)
		} else {
			for _, st := range out.stepsForValue(i) {
				if st.kind == abiStepPointer {
					outRegPtrs.Set(st.ireg)
				}
			}
		}
	}
	// Undo the faking from earlier so that stackBytes
	// is accurate.
	out.stackBytes -= retOffset
	return abiDesc{in, out, stackCallArgsSize, retOffset, spill, stackPtrs, inRegPtrs, outRegPtrs}
}

// intFromReg loads an argSize sized integer from reg and places it at to.
//
// argSize must be non-zero, fit in a register, and a power-of-two.
func intFromReg(r *abi.RegArgs, reg int, argSize uintptr, to unsafe.Pointer) {
	memmove(to, r.IntRegArgAddr(reg, argSize), argSize)
}

// intToReg loads an argSize sized integer and stores it into reg.
//
// argSize must be non-zero, fit in a register, and a power-of-two.
func intToReg(r *abi.RegArgs, reg int, argSize uintptr, from unsafe.Pointer) {
	memmove(r.IntRegArgAddr(reg, argSize), from, argSize)
}

// floatFromReg loads a float value from its register representation in r.
//
// argSize must be 4 or 8.
func floatFromReg(r *abi.RegArgs, reg int, argSize uintptr, to unsafe.Pointer) {
	switch argSize {
	case 4:
		*(*float32)(to) = archFloat32FromReg(r.Floats[reg])
	case 8:
		*(*float64)(to) = *(*float64)(unsafe.Pointer(&r.Floats[reg]))
	default:
		panic("bad argSize")
	}
}

// floatToReg stores a float value in its register representation in r.
//
// argSize must be either 4 or 8.
func floatToReg(r *abi.RegArgs, reg int, argSize uintptr, from unsafe.Pointer) {
	switch argSize {
	case 4:
		r.Floats[reg] = archFloat32ToReg(*(*float32)(from))
	case 8:
		r.Floats[reg] = *(*uint64)(from)
	default:
		panic("bad argSize")
	}
}
