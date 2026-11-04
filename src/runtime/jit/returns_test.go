// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit && (amd64 || arm64 || riscv64) && (unix || windows)

package jit_test

import (
	"encoding/binary"
	"math"
	"reflect"
	"runtime"
	"runtime/jit"
	"testing"
	"unsafe"
)

// These are deliberately hand-encoded rather than emitted by runtime/jit's
// instruction helpers. A toolchain ABI change must update the ABI description
// and its code emitters; describing and emitting the same wrong convention must
// not make the tests pass. Calls originate in normally compiled Go code.
func emitConstantReturn(b *jit.SliceBuffer, result jit.PhysicalRegister, bits uint64) {
	word := func(v uint32) { var code [4]byte; binary.LittleEndian.PutUint32(code[:], v); b.Write(code[:]) }
	integer := result.Number
	if result.Class == jit.FloatRegister {
		integer = jit.RegisterConfig().Scratch.Number
	}
	switch runtime.GOARCH {
	case "amd64":
		code := []byte{0x48 | integer>>3, 0xb8 | integer&7, 0, 0, 0, 0, 0, 0, 0, 0}
		binary.LittleEndian.PutUint64(code[2:], bits)
		b.Write(code)
		if result.Class == jit.FloatRegister {
			b.Write([]byte{0x66, 0x48 | result.Number>>3<<2 | integer>>3, 0x0f, 0x6e, 0xc0 | result.Number&7<<3 | integer&7})
		}
	case "arm64":
		word(0xd2800000 | uint32(bits&65535)<<5 | uint32(integer))
		for i := uint32(1); i < 4; i++ {
			word(0xf2800000 | i<<21 | uint32(bits>>(16*i)&65535)<<5 | uint32(integer))
		}
		if result.Class == jit.FloatRegister {
			word(0x9e670000 | uint32(integer)<<5 | uint32(result.Number))
		}
	case "riscv64":
		// AUIPC/LD/JAL around a complete 64-bit inline constant.
		word(uint32(integer)<<7 | 0x17)
		word(16<<20 | uint32(integer)<<15 | 3<<12 | uint32(integer)<<7 | 3)
		word(8<<21 | 0x6f)
		word(0x13)
		var literal [8]byte
		binary.LittleEndian.PutUint64(literal[:], bits)
		b.Write(literal[:])
		if result.Class == jit.FloatRegister {
			word(0xf2000053 | uint32(integer)<<15 | uint32(result.Number)<<7)
		}
	}
	jit.EmitLeafReturn(b)
}

func registerReturnCode(t *testing.T, b *jit.SliceBuffer, maps []jit.StackMap) (uintptr, func()) {
	t.Helper()
	addr, size, err := allocExecutable(b.Bytes[:b.Offset()])
	if err != nil {
		t.Fatal(err)
	}
	h := jit.Register(jit.Region{Start: addr, End: addr + uintptr(size), Unwind: jit.UnwindSkip, StackMaps: maps})
	return addr, func() { h.Unregister(); freeExecutable(addr, size) }
}

func TestABIHandEmittedIntegerReturn(t *testing.T) {
	c := jit.GetCallingConventionForFunc((func() int64)(nil))
	b := &jit.SliceBuffer{Bytes: make([]byte, 128)}
	emitConstantReturn(b, c.Outputs[0].Parts[0].Location.Register, 0x123456789abcdef)
	addr, cleanup := registerReturnCode(t, b, nil)
	defer cleanup()
	h := new(jit.FunctionHeader)
	h.SetEntry(addr)
	if got := jit.Function[func() int64](h)(); got != 0x123456789abcdef {
		t.Fatalf("Go ABI integer return=%#x", got)
	}
}

func TestABIHandEmittedFloatReturn(t *testing.T) {
	c := jit.GetCallingConventionForFunc((func() float64)(nil))
	b := &jit.SliceBuffer{Bytes: make([]byte, 128)}
	emitConstantReturn(b, c.Outputs[0].Parts[0].Location.Register, math.Float64bits(42.125))
	addr, cleanup := registerReturnCode(t, b, nil)
	defer cleanup()
	h := new(jit.FunctionHeader)
	h.SetEntry(addr)
	if got := jit.Function[func() float64](h)(); got != 42.125 {
		t.Fatalf("Go ABI float return=%v", got)
	}
}

// This independently encodes the context load. A closure calling convention
// change must not be hidden by changing both the descriptor and production
// emitter in the same (incorrect) way.
func emitContextReturn(b *jit.SliceBuffer, result jit.PhysicalRegister, offset uintptr) {
	context := jit.RegisterConfig().Closure.Number
	word := func(v uint32) { var code [4]byte; binary.LittleEndian.PutUint32(code[:], v); b.Write(code[:]) }
	switch runtime.GOARCH {
	case "amd64":
		if result.Class == jit.FloatRegister {
			b.Write([]byte{0xf2})
		}
		rex := byte(0x40) | result.Number>>3<<2 | context>>3
		if result.Class == jit.IntegerRegister {
			rex |= 8
		}
		b.Write([]byte{rex})
		if result.Class == jit.FloatRegister {
			b.Write([]byte{0x0f, 0x10})
		} else {
			b.Write([]byte{0x8b})
		}
		b.Write([]byte{0x80 | result.Number&7<<3 | context&7})
		if context&7 == 4 {
			b.Write([]byte{0x24})
		}
		word(uint32(offset))
	case "arm64":
		op := uint32(0xf9400000)
		if result.Class == jit.FloatRegister {
			op = 0xfd400000
		}
		word(op | uint32(offset/8)<<10 | uint32(context)<<5 | uint32(result.Number))
	case "riscv64":
		op := uint32(3)
		if result.Class == jit.FloatRegister {
			op = 7
		}
		word(uint32(offset)<<20 | uint32(context)<<15 | 3<<12 | uint32(result.Number)<<7 | op)
	}
	jit.EmitLeafReturn(b)
}

func TestABIHandEmittedClosureReturns(t *testing.T) {
	t.Run("integer", func(t *testing.T) {
		type context struct {
			jit.FunctionHeader
			value int64
		}
		owner := &context{value: 0x123456789abcdef}
		c := jit.GetCallingConventionForFunc((func() int64)(nil))
		b := &jit.SliceBuffer{Bytes: make([]byte, 128)}
		emitContextReturn(b, c.Outputs[0].Parts[0].Location.Register, unsafe.Offsetof(owner.value))
		addr, cleanup := registerReturnCode(t, b, nil)
		defer cleanup()
		owner.SetEntry(addr)
		fn := jit.Function[func() int64](&owner.FunctionHeader)
		if got := fn(); got != owner.value {
			t.Fatalf("closure integer=%#x", got)
		}
		owner.value = -17 // Prove the generated function loads rather than embeds it.
		if got := fn(); got != -17 {
			t.Fatalf("updated closure integer=%d", got)
		}
	})
	t.Run("float", func(t *testing.T) {
		type context struct {
			jit.FunctionHeader
			value float64
		}
		owner := &context{value: 42.125}
		c := jit.GetCallingConventionForFunc((func() float64)(nil))
		b := &jit.SliceBuffer{Bytes: make([]byte, 128)}
		emitContextReturn(b, c.Outputs[0].Parts[0].Location.Register, unsafe.Offsetof(owner.value))
		addr, cleanup := registerReturnCode(t, b, nil)
		defer cleanup()
		owner.SetEntry(addr)
		fn := jit.Function[func() float64](&owner.FunctionHeader)
		if got := fn(); got != owner.value {
			t.Fatalf("closure float=%v", got)
		}
		owner.value = -17.5
		if got := fn(); got != -17.5 {
			t.Fatalf("updated closure float=%v", got)
		}
	})
}

//go:noinline
func integerReturnCallee() int64 { runtime.GC(); return 0x123456789abcdef }

//go:noinline
func floatReturnCallee() float64 { runtime.GC(); return 42.125 }

func TestABIEmittedGoCallReturns(t *testing.T) {
	for _, fn := range []any{integerReturnCallee, floatReturnCallee} {
		t.Run(reflect.TypeOf(fn).String(), func(t *testing.T) {
			c := jit.GetCallingConventionForFunc(fn)
			b := &jit.SliceBuffer{Bytes: make([]byte, 512)}
			frameBytes := c.CallFrameSize + 16
			jit.EmitFrame(b, jit.FramePlan{LocalBytes: int32(frameBytes)})
			// A chosen JIT frame register can be caller-clobbered (notably on
			// riscv64). Preserve it just like any other live register across Go.
			emitTestFrameSpill(b, false, c.CallFrameSize)
			call := jit.EmitCall(b, jit.CallPlan{Entry: reflect.ValueOf(fn).Pointer()})
			emitTestFrameSpill(b, true, c.CallFrameSize)
			jit.EmitReturn(b, c.Registers.Frame)
			mask := make([]byte, (frameBytes/8+1+7)/8)
			word := c.CallFrameSize / 8
			mask[word/8] |= 1 << uint(word%8)
			m := jit.FrameStackMap(uintptr(call.ReturnPC), frameBytes, mask)
			addr, cleanup := registerReturnCode(t, b, []jit.StackMap{m})
			defer cleanup()
			h := new(jit.FunctionHeader)
			h.SetEntry(addr)
			switch fn.(type) {
			case func() int64:
				if got := jit.Function[func() int64](h)(); got != integerReturnCallee() {
					t.Fatalf("integer call returned %#x", got)
				}
			case func() float64:
				if got := jit.Function[func() float64](h)(); got != floatReturnCallee() {
					t.Fatalf("float call returned %v", got)
				}
			}
		})
	}
}

func emitTestFrameSpill(b *jit.SliceBuffer, load bool, offset uintptr) {
	r := jit.RegisterConfig()
	frame, sp := uint32(r.Frame.Number), uint32(r.Stack.Number)
	word := func(v uint32) { var code [4]byte; binary.LittleEndian.PutUint32(code[:], v); b.Write(code[:]) }
	switch runtime.GOARCH {
	case "amd64":
		op := byte(0x89)
		if load {
			op = 0x8b
		}
		b.Write([]byte{0x48 | byte(frame>>3)<<2 | byte(sp>>3), op, 0x84 | byte(frame&7)<<3, 0x24})
		word(uint32(offset))
	case "arm64":
		op := uint32(0xf9000000)
		if load {
			op = 0xf9400000
		}
		word(op | uint32(offset/8)<<10 | sp<<5 | frame)
	case "riscv64":
		v := uint32(offset)
		if load {
			word(v<<20 | sp<<15 | 3<<12 | frame<<7 | 3)
		} else {
			word(v>>5<<25 | frame<<20 | sp<<15 | 3<<12 | (v&31)<<7 | 0x23)
		}
	}
}

// A large consumer function can place its growth path beyond JAL's reach.
// Exercise the emitted forward check and backward retry, rather than merely
// accepting a large displacement during encoding.
func TestEmissionLargeRiscvGrowth(t *testing.T) {
	if runtime.GOARCH != "riscv64" {
		t.Skip("RISC-V far growth branches")
	}
	c := jit.GetCallingConventionForFunc((func([]byte) uintptr)(nil))
	b := &jit.SliceBuffer{Bytes: make([]byte, 2<<20)}
	check := jit.EmitStackCheck(b, 0, 0)
	jit.EmitFrame(b, jit.FramePlan{LocalBytes: 32768})
	result, input := c.Outputs[0].Parts[0].Location.Register.Number, c.Inputs[0].Parts[1].Location.Register.Number
	var move [4]byte
	binary.LittleEndian.PutUint32(move[:], uint32(input)<<15|uint32(result)<<7|0x13)
	b.Write(move[:])
	jit.EmitReturn(b, c.Registers.Frame)
	b.Write(make([]byte, (1<<20)+4096-b.Offset())) // Unreachable padding.
	grow := b.Offset()
	m := jit.EmitStackGrow(b, c, c.Registers.Frame, 0)
	jit.PatchStackCheck(b, check, 32768+16, grow)
	addr, cleanup := registerReturnCode(t, b, []jit.StackMap{m})
	defer cleanup()
	h := new(jit.FunctionHeader)
	h.SetEntry(addr)
	fn := jit.Function[func([]byte) uintptr](h)
	done := make(chan uintptr, 1)
	go func() { done <- fn(make([]byte, 123)) }()
	if got := <-done; got != 123 {
		t.Fatalf("argument lost across far growth path: %d", got)
	}
}
