// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit && amd64 && (unix || windows)

package jit_test

import (
	"reflect"
	"runtime"
	"runtime/jit"
	"sync/atomic"
	"testing"
)

//go:noinline
func abiMixedCallee(i uintptr, f float64, p *int) (uintptr, float64, *int) {
	runtime.GC()
	return i + 1, f + 0.5, p
}

func executableBuffer(t *testing.T, emit func(*jit.SliceBuffer) []jit.StackMap) (uintptr, func()) {
	t.Helper()
	b := &jit.SliceBuffer{Bytes: make([]byte, 4096)}
	maps := emit(b)
	addr, size, err := allocExecutable(b.Bytes[:b.Position])
	if err != nil {
		t.Fatal(err)
	}
	h := jit.Register(jit.Region{Start: addr, End: addr + uintptr(size), Unwind: jit.UnwindSkip, StackMaps: maps})
	return addr, func() { h.Unregister(); freeExecutable(addr, size) }
}

func TestEmittedGoCallMixedRegisters(t *testing.T) {
	c := jit.GetCallingConventionForFunc(abiMixedCallee)
	addr, cleanup := executableBuffer(t, func(b *jit.SliceBuffer) []jit.StackMap {
		jit.EmitFrame(b, jit.FramePlan{LocalBytes: int32(c.CallFrameSize)})
		call := jit.EmitCall(b, jit.CallPlan{Entry: reflect.ValueOf(abiMixedCallee).Pointer()})
		jit.EmitReturn(b, c.Registers.Frame)
		return []jit.StackMap{jit.FrameStackMap(uintptr(call.ReturnPC), c.CallFrameSize, make([]byte, (c.CallFrameSize/8+1+7)/8))}
	})
	defer cleanup()
	h := new(jit.FunctionHeader)
	h.SetEntry(addr)
	fn := jit.Function[func(uintptr, float64, *int) (uintptr, float64, *int)](h)
	v := 19
	a, f, p := fn(41, 2.25, &v)
	if a != 42 || f != 2.75 || p != &v || *p != 19 {
		t.Fatalf("bad mixed result: %d %f %p", a, f, p)
	}
	runtime.KeepAlive(h)
}

func TestEmittedClosureCall(t *testing.T) {
	captured := new(int)
	*captured = 17
	closure := func(v uintptr) uintptr { runtime.GC(); return v + uintptr(*captured) }
	context := jit.FunctionContext(closure)
	c := jit.GetCallingConventionForFunc(closure)
	addr, cleanup := executableBuffer(t, func(b *jit.SliceBuffer) []jit.StackMap {
		jit.EmitFrame(b, jit.FramePlan{LocalBytes: int32(c.CallFrameSize)})
		// Test-only materialization: the Go closure itself remains rooted below.
		code := []byte{0x48, 0xba, 0, 0, 0, 0, 0, 0, 0, 0}
		putU64(code[2:], uintptr(context))
		b.Write(code)
		r := c.Registers.Closure
		call := jit.EmitCall(b, jit.CallPlan{Closure: &r})
		jit.EmitReturn(b, c.Registers.Frame)
		return []jit.StackMap{jit.FrameStackMap(uintptr(call.ReturnPC), c.CallFrameSize, make([]byte, (c.CallFrameSize/8+1+7)/8))}
	})
	defer cleanup()
	h := new(jit.FunctionHeader)
	h.SetEntry(addr)
	if got := jit.Function[func(uintptr) uintptr](h)(25); got != 42 {
		t.Fatalf("closure returned %d", got)
	}
	runtime.KeepAlive(closure)
	runtime.KeepAlive(context)
}

func TestEmittedFramePatchAndStackGrowth(t *testing.T) {
	c := jit.GetCallingConventionForFunc((func([]byte) uintptr)(nil))
	addr, cleanup := executableBuffer(t, func(b *jit.SliceBuffer) []jit.StackMap {
		check := jit.EmitStackCheck(b, 0, 0)
		frame := jit.EmitFrame(b, jit.FramePlan{})
		// A 32KB frame forces the generated slow path on a fresh goroutine.
		jit.PatchFrame(b, frame, 32768)
		// Return slice length, exercising argument preservation across growth.
		b.Write([]byte{0x48, 0x89, 0xd8})
		jit.EmitReturn(b, c.Registers.Frame)
		grow := b.Offset()
		m := jit.EmitStackGrow(b, c, c.Registers.Frame, 0)
		jit.PatchStackCheck(b, check, 32768+8, grow)
		return []jit.StackMap{m}
	})
	defer cleanup()
	h := new(jit.FunctionHeader)
	h.SetEntry(addr)
	fn := jit.Function[func([]byte) uintptr](h)
	result := make(chan uintptr, 1)
	go func() { result <- fn(make([]byte, 123)) }()
	if got := <-result; got != 123 {
		t.Fatalf("argument lost across stack growth: %d", got)
	}
}

func TestEmittedLoopCooperatesWithSchedulerAndGC(t *testing.T) {
	// No ordinary Go callbacks on the loop back-edge. Poll g.stackguard0;
	// negative sentinel values request preemption. Yield only on that request.
	r := jit.RegisterConfig()
	guard := jit.StackCheckConfig().StackGuardOffset
	addr, cleanup := executableBuffer(t, func(b *jit.SliceBuffer) []jit.StackMap {
		jit.EmitFrame(b, jit.FramePlan{})
		loop := b.Offset()
		// MOV RAX,[RDX+8] loads a typed atomic stop flag rooted by the funcval.
		b.Write([]byte{0x48, 0x8b, 0x42, 0x08, 0x83, 0x38, 0x00, 0x0f, 0x85, 0, 0, 0, 0})
		stopPatch := b.Offset() - 4
		code := []byte{0x49, 0x8b, 0x86, 0, 0, 0, 0, 0x48, 0x85, 0xc0, 0x0f, 0x89, 0, 0, 0, 0}
		put32 := func(p []byte, v int32) {
			p[0] = byte(v)
			p[1] = byte(v >> 8)
			p[2] = byte(v >> 16)
			p[3] = byte(v >> 24)
		}
		put32(code[3:], int32(guard))
		put32(code[12:], int32(loop-(b.Offset()+len(code))))
		b.Write(code)
		// Preserve closure context as a precise root across the scheduling call.
		b.Write([]byte{0x52, 0x52})
		call := jit.EmitCall(b, jit.CallPlan{Entry: reflect.ValueOf(jit.Yield).Pointer()})
		b.Write([]byte{0x5a, 0x5a, 0xe9, 0, 0, 0, 0})
		patch := make([]byte, 4)
		put32(patch, int32(loop-b.Offset()))
		b.Patch(b.Offset()-4, patch)
		stop := b.Offset()
		put32(patch, int32(stop-stopPatch-4))
		b.Patch(stopPatch, patch)
		jit.EmitReturn(b, r.Frame)
		return []jit.StackMap{jit.FrameStackMap(uintptr(call.ReturnPC), 16, []byte{3})}
	})
	defer cleanup()
	var flag uint32
	h := &struct {
		jit.FunctionHeader
		stop *uint32
	}{stop: &flag}
	h.SetEntry(addr)
	fn := jit.Function[func()](&h.FunctionHeader)
	old := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(old)
	done := make(chan struct{})
	go func() { fn(); close(done) }()
	// Scheduling must run this goroutine and GC despite the native loop.
	runtime.Gosched()
	runtime.GC()
	// Go and generated code use atomic loads/stores on amd64.
	atomicStoreStop(&flag)
	<-done
	runtime.KeepAlive(h)
}

func atomicStoreStop(p *uint32) { atomic.StoreUint32(p, 1) }
