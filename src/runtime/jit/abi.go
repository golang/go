// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit

package jit

import (
	"internal/abi"
	"internal/goarch"
	"reflect"
	"runtime"
)

// RegisterClass identifies a physical register file. PhysicalRegister numbers are
// architectural encodings within that file, not compiler register colors.
type RegisterClass uint8

const (
	IntegerRegister RegisterClass = iota
	FloatRegister
)

// PhysicalRegister names a physical register in the running toolchain's architecture.
type PhysicalRegister struct {
	Class  RegisterClass
	Number uint8
}

// RegisterSet is indexed by RegisterClass, then by architectural register number.
type RegisterSet [2]uint64

// Contains reports whether r belongs to s.
func (s RegisterSet) Contains(r PhysicalRegister) bool { return s[r.Class]&(uint64(1)<<r.Number) != 0 }

// Registers describes ABI constraints, not a client's dynamic allocation state.
// A register used for an argument is constrained at that boundary only.
type Registers struct {
	IntegerArguments                          []PhysicalRegister
	FloatArguments                            []PhysicalRegister
	Allocatable                               RegisterSet
	Reserved                                  RegisterSet
	CallClobbers                              RegisterSet
	Stack, Frame, Goroutine, Closure, Scratch PhysicalRegister
	StackAlignment                            uintptr
	// StackArgumentOffset is the minimum caller-owned linkage area at caller SP.
	StackArgumentOffset uintptr
	// EntrySPDelta is subtracted from caller SP by the CALL instruction.
	EntrySPDelta uintptr
}

// RegisterConfig returns constraints for this running toolchain. Returned slices
// belong to the caller. Code and these descriptions must never be persisted for
// use with another architecture or toolchain version.
func RegisterConfig() Registers {
	r := nativeRegisterConfig()
	r.IntegerArguments = append([]PhysicalRegister(nil), r.IntegerArguments...)
	r.FloatArguments = append([]PhysicalRegister(nil), r.FloatArguments...)
	return r
}

var nativeABI = func() Registers {
	if !SupportedEmitters() {
		return Registers{}
	}
	return registerConfig()
}()

func nativeRegisterConfig() Registers {
	if !SupportedEmitters() {
		panic("runtime/jit: ABI helpers unsupported on " + runtime.GOARCH)
	}
	return nativeABI
}

func registerConfig() Registers {
	r := Registers{StackAlignment: 16}
	ints := []uint8(nil)
	floats := abi.FloatArgRegs
	switch runtime.GOARCH {
	case "amd64":
		ints = []uint8{0, 3, 1, 7, 6, 8, 9, 10, 11}
		r.Stack, r.Frame, r.Goroutine, r.Closure, r.Scratch = gp(4), gp(5), gp(14), gp(2), gp(12)
		r.Allocatable = RegisterSet{0xffff &^ (1<<4 | 1<<5 | 1<<14), 0xffff}
		r.EntrySPDelta = 8
	case "arm64":
		for i := uint8(0); i < 16; i++ {
			ints = append(ints, i)
		}
		// Reuse native instruction temporary X17 outside the Go argument bank.
		// X27 remains available to clients as a call-clobbered temporary.
		r.Stack, r.Frame, r.Goroutine, r.Closure, r.Scratch = gp(31), gp(29), gp(28), gp(26), gp(17)
		r.Allocatable = RegisterSet{(uint64(1)<<31 - 1) &^ (1<<18 | 1<<28 | 1<<29 | 1<<30), uint64(1)<<32 - 1}
		r.StackArgumentOffset = 8
	case "riscv64":
		ints = []uint8{10, 11, 12, 13, 14, 15, 16, 17, 8, 9, 18, 19, 20, 21, 22, 23}
		r.Stack, r.Frame, r.Goroutine, r.Closure, r.Scratch = gp(2), gp(24), gp(27), gp(26), gp(31)
		r.Allocatable = RegisterSet{(uint64(1)<<32 - 1) &^ (1<<0 | 1<<1 | 1<<2 | 1<<3 | 1<<4 | 1<<27), uint64(1)<<32 - 1}
		r.StackArgumentOffset = 8
	default:
		panic("runtime/jit: ABI helpers unsupported on " + runtime.GOARCH)
	}
	for _, n := range ints {
		r.IntegerArguments = append(r.IntegerArguments, gp(n))
	}
	for i := 0; i < floats; i++ {
		n := uint8(i)
		if runtime.GOARCH == "riscv64" {
			n = []uint8{10, 11, 12, 13, 14, 15, 16, 17, 8, 9, 18, 19, 20, 21, 22, 23}[i]
		}
		r.FloatArguments = append(r.FloatArguments, PhysicalRegister{FloatRegister, n})
	}
	r.CallClobbers = r.Allocatable
	// Toolchain scratch registers are not safe persistent homes, even though
	// generated instruction sequences may use them explicitly.
	r.Allocatable[0] &^= uint64(1)<<r.Scratch.Number | uint64(1)<<r.Frame.Number
	if runtime.GOARCH == "amd64" {
		// X15 must contain zero at Go call/return boundaries. R15, in contrast,
		// is only implicit scratch for the Go assembler's dynlink expansions.
		// These emitters do not use those expansions: clients may allocate R15
		// and preserve it according to CallClobbers, just like R13.
		r.Allocatable[1] &^= 1 << 15
	}
	for i := range r.Reserved {
		r.Reserved[i] = ^r.Allocatable[i]
	}
	return r
}

func gp(n uint8) PhysicalRegister { return PhysicalRegister{IntegerRegister, n} }

// LocationKind distinguishes a register from a caller-SP-relative stack slot.
type LocationKind uint8

const (
	InRegister LocationKind = iota
	OnStack
)

// Location describes a machine location at a call boundary. StackOffset is
// relative to caller SP immediately before CALL; at function entry add
// Registers.EntrySPDelta. It is never relative to a client's local frame base.
type Location struct {
	Kind        LocationKind
	Register    PhysicalRegister
	StackOffset uintptr
}

// ValuePart maps a portion of a Go value onto an ABI location. Pointer means a
// relocatable GC pointer. Empty-interface values currently consist of a type
// metadata word and a data pointer word; only the latter is a GC pointer here.
// The data word is not an arbitrary unboxed integer. A nonempty interface uses
// an itab instead of a type word. Clients inspecting these words depend on this
// representation; calling-convention queries only abstract their placement.
type ValuePart struct {
	ValueOffset, Size uintptr
	Pointer           bool
	Location          Location
}

// ValueLayout preserves one original parameter/result, including empty values.
type ValueLayout struct {
	Type  reflect.Type
	Parts []ValuePart
	// SpillOffset is the caller-SP-relative register-argument spill home.
	// HasSpill is false for stack arguments and all results.
	SpillOffset uintptr
	HasSpill    bool
}

// CallingConvention describes ABIInternal for a Go function signature in the
// running toolchain. Variadic arguments are represented by their final slice.
// It is not a stable binary ABI and cannot be reused across toolchain versions.
type CallingConvention struct {
	Inputs, Outputs                                          []ValueLayout
	Registers                                                Registers
	StackArgumentsSize, StackResultsOffset, StackResultsSize uintptr
	SpillOffset, SpillSize, CallFrameSize                    uintptr
}

// GetCallingConventionForFunc accepts a Go function value (including a typed
// nil). For a method expression the receiver is an ordinary first parameter;
// a bound method's receiver is held by its closure context.
func GetCallingConventionForFunc(fn any) CallingConvention {
	return GetCallingConventionForType(reflect.TypeOf(fn))
}

// GetCallingConventionForType also supports signatures whose JIT function value
// does not exist yet. It panics if t is not a function type.
func GetCallingConventionForType(t reflect.Type) CallingConvention {
	if t == nil || t.Kind() != reflect.Func {
		panic("runtime/jit: expected function type")
	}
	c := CallingConvention{Registers: RegisterConfig()}
	stack := c.Registers.StackArgumentOffset
	assign := func(count int, typ func(int) reflect.Type, input bool) []ValueLayout {
		layouts := make([]ValueLayout, count)
		ni, nf := 0, 0
		for i := range layouts {
			t := typ(i)
			v := ValueLayout{Type: t}
			parts, ok := registerParts(t, 0)
			oldI, oldF := ni, nf
			if t.Size() == 0 {
				ok = false
			}
			for j := range parts {
				p := &parts[j]
				bank, index := c.Registers.IntegerArguments, &ni
				if p.Location.Register.Class == FloatRegister {
					bank, index = c.Registers.FloatArguments, &nf
				}
				if *index >= len(bank) {
					ok = false
					break
				}
				p.Location = Location{Kind: InRegister, Register: bank[*index]}
				*index++
			}
			if ok {
				v.Parts = parts
				v.HasSpill = input
			} else {
				ni, nf = oldI, oldF
				stack = alignABI(stack, uintptr(t.Align()))
				v.Parts = stackParts(t, 0, stack)
				stack += t.Size()
			}
			layouts[i] = v
		}
		return layouts
	}
	c.Inputs = assign(t.NumIn(), t.In, true)
	stack = alignABI(stack, goarch.PtrSize)
	c.StackArgumentsSize = stack - c.Registers.StackArgumentOffset
	c.StackResultsOffset = stack
	c.Outputs = assign(t.NumOut(), t.Out, false)
	stack = alignABI(stack, goarch.PtrSize)
	c.StackResultsSize = stack - c.StackResultsOffset
	c.SpillOffset = stack
	for i := range c.Inputs {
		v := &c.Inputs[i]
		if !v.HasSpill {
			continue
		}
		stack = alignABI(stack, uintptr(v.Type.Align()))
		v.SpillOffset = stack
		stack += v.Type.Size()
	}
	c.SpillSize = alignABI(stack, goarch.PtrSize) - c.SpillOffset
	c.CallFrameSize = alignABI(c.SpillOffset+c.SpillSize, c.Registers.StackAlignment)
	return c
}

func alignABI(n, a uintptr) uintptr { return (n + a - 1) &^ (a - 1) }

func registerParts(t reflect.Type, off uintptr) ([]ValuePart, bool) {
	part := func(size uintptr, pointer bool, class RegisterClass) ValuePart {
		return ValuePart{ValueOffset: off, Size: size, Pointer: pointer, Location: Location{Register: PhysicalRegister{Class: class}}}
	}
	switch t.Kind() {
	case reflect.Array:
		if t.Len() == 0 {
			return nil, true
		}
		if t.Len() == 1 {
			return registerParts(t.Elem(), off)
		}
		return nil, false
	case reflect.Struct:
		var out []ValuePart
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			p, ok := registerParts(f.Type, off+f.Offset)
			if !ok {
				return nil, false
			}
			out = append(out, p...)
		}
		return out, true
	case reflect.Float32, reflect.Float64:
		return []ValuePart{part(t.Size(), false, FloatRegister)}, true
	case reflect.Complex64, reflect.Complex128:
		p := part(t.Size()/2, false, FloatRegister)
		q := p
		q.ValueOffset += p.Size
		return []ValuePart{p, q}, true
	case reflect.String, reflect.Slice, reflect.Interface:
		n := 2
		if t.Kind() == reflect.Slice {
			n = 3
		}
		out := make([]ValuePart, n)
		for i := range out {
			pointer := i == 0
			if t.Kind() == reflect.Interface {
				pointer = i == 1
			}
			out[i] = part(goarch.PtrSize, pointer, IntegerRegister)
			out[i].ValueOffset += uintptr(i) * goarch.PtrSize
		}
		return out, true
	case reflect.Pointer, reflect.UnsafePointer, reflect.Map, reflect.Chan, reflect.Func:
		return []ValuePart{part(t.Size(), true, IntegerRegister)}, true
	default:
		if t.Size() > goarch.PtrSize {
			p := part(goarch.PtrSize, false, IntegerRegister)
			q := p
			q.ValueOffset += goarch.PtrSize
			return []ValuePart{p, q}, true
		}
		return []ValuePart{part(t.Size(), false, IntegerRegister)}, true
	}
}

func stackParts(t reflect.Type, off, base uintptr) []ValuePart {
	if t.Kind() == reflect.Array {
		var out []ValuePart
		for i := 0; i < t.Len(); i++ {
			out = append(out, stackParts(t.Elem(), off+uintptr(i)*t.Elem().Size(), base)...)
		}
		return out
	}
	if t.Kind() == reflect.Struct {
		var out []ValuePart
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			out = append(out, stackParts(f.Type, off+f.Offset, base)...)
		}
		return out
	}
	if t.Size() == 0 {
		return nil
	}
	parts, _ := registerParts(t, off)
	for i := range parts {
		parts[i].Location = Location{Kind: OnStack, StackOffset: base + parts[i].ValueOffset}
	}
	return parts
}
