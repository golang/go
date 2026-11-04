// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit

package jit_test

import (
	"bytes"
	"errors"
	"reflect"
	"runtime"
	"runtime/jit"
	"testing"
)

func TestCallingConvention(t *testing.T) {
	if !jit.SupportedEmitters() {
		t.Skip("ABI emitter target")
	}
	type mixed struct {
		I uintptr
		F float64
		P *byte
	}
	c := jit.GetCallingConventionForFunc((func(mixed, any, []byte) (complex128, *int))(nil))
	if len(c.Inputs) != 3 || len(c.Outputs) != 2 || len(c.Inputs[0].Parts) != 3 {
		t.Fatalf("lost parameter boundaries: %+v", c)
	}
	if c.Inputs[0].Parts[1].Location.Register.Class != jit.FloatRegister || !c.Inputs[0].Parts[2].Pointer {
		t.Fatal("mixed aggregate layout")
	}
	if c.Inputs[1].Parts[0].Pointer || !c.Inputs[1].Parts[1].Pointer {
		t.Fatal("interface pointer roles")
	}
	if c.Outputs[0].Parts[0].Location.Register != c.Registers.FloatArguments[0] {
		t.Fatal("result register bank did not restart")
	}
	if c.CallFrameSize%16 != 0 || c.SpillSize == 0 {
		t.Fatal("missing aligned spill area")
	}
	if c.Registers.Allocatable.Contains(c.Registers.Goroutine) || c.Registers.Allocatable.Contains(c.Registers.Stack) {
		t.Fatal("reserved register allocatable")
	}
	if runtime.GOARCH == "amd64" {
		if !c.Registers.Allocatable.Contains(jit.PhysicalRegister{Class: jit.IntegerRegister, Number: 15}) {
			t.Fatal("R15 is usable by direct emitters and call-clobbered, not a fixed register")
		}
		if c.Registers.Allocatable.Contains(jit.PhysicalRegister{Class: jit.FloatRegister, Number: 15}) {
			t.Fatal("X15 must remain zero at Go boundaries")
		}
	}
}

func TestMemoryLayoutAggregatesAndInterfaces(t *testing.T) {
	type object struct {
		Tag       byte
		Values    [2]*int
		Interface any
		Slice     []byte
	}
	typ := reflect.TypeFor[object]()
	m := jit.GetMemoryLayout(typ)
	if m.Size != typ.Size() || m.Align != uintptr(typ.Align()) {
		t.Fatal("incorrect aggregate extent")
	}
	var pointers []uintptr
	for _, part := range m.Parts {
		if part.Pointer {
			pointers = append(pointers, part.Offset)
		}
	}
	word := reflect.TypeFor[uintptr]().Size()
	want := []uintptr{typ.Field(1).Offset, typ.Field(1).Offset + word, typ.Field(2).Offset + word, typ.Field(3).Offset}
	if !reflect.DeepEqual(pointers, want) {
		t.Fatalf("pointer offsets %v, want %v", pointers, want)
	}
	// Returned descriptions must not share mutable storage across consumers.
	m.Parts[0].Offset = 1234
	if jit.GetMemoryLayout(typ).Parts[0].Offset != 0 {
		t.Fatal("shared layout storage")
	}
}

func TestCallingConventionAggregateRollback(t *testing.T) {
	if !jit.SupportedEmitters() {
		t.Skip("ABI emitter target")
	}
	r := jit.RegisterConfig()
	args := make([]reflect.Type, len(r.IntegerArguments)-1)
	for i := range args {
		args[i] = reflect.TypeFor[uintptr]()
	}
	args = append(args, reflect.TypeFor[[]byte](), reflect.TypeFor[uintptr](), reflect.TypeFor[[2]*int](), reflect.TypeFor[struct{}]())
	c := jit.GetCallingConventionForType(reflect.FuncOf(args, []reflect.Type{reflect.TypeFor[[2]uintptr]()}, false))
	slice := c.Inputs[len(r.IntegerArguments)-1]
	for _, p := range slice.Parts {
		if p.Location.Kind != jit.OnStack {
			t.Fatal("aggregate split across registers and stack")
		}
	}
	last := c.Inputs[len(r.IntegerArguments)]
	if last.Parts[0].Location.Register != r.IntegerArguments[len(r.IntegerArguments)-1] {
		t.Fatal("failed aggregate consumed registers")
	}
	array := c.Inputs[len(r.IntegerArguments)+1]
	if len(array.Parts) != 2 || !array.Parts[0].Pointer || array.Parts[0].Location.Kind != jit.OnStack {
		t.Fatal("array stack layout")
	}
	if c.Outputs[0].Parts[0].Location.StackOffset != c.StackResultsOffset {
		t.Fatal("result stack offset")
	}
}

func TestCodeBufferFullIsAtomic(t *testing.T) {
	b := &jit.SliceBuffer{Bytes: make([]byte, 3)}
	b.Write([]byte{1, 2})
	func() {
		defer func() {
			v := recover()
			e, ok := v.(jit.CodeBufferFull)
			if !ok || e.Offset != 2 || e.Required != 2 || e.Available != 1 {
				t.Fatalf("overflow: %#v", v)
			}
		}()
		b.Write([]byte{3, 4})
		t.Fatal("missing panic")
	}()
	if b.Offset() != 2 || !bytes.Equal(b.Bytes, []byte{1, 2, 0}) {
		t.Fatal("partial write on exhaustion")
	}
	b.Patch(0, []byte{9})
	if b.Offset() != 2 || b.Bytes[0] != 9 {
		t.Fatal("patch moved cursor")
	}
	func() {
		defer func() {
			if recover() == nil {
				t.Error("patch outside written bytes accepted")
			}
		}()
		b.Patch(2, []byte{1})
	}()
}

func TestMetadataValidationAndSnapshot(t *testing.T) {
	r := jit.Region{Start: 0x1000, End: 0x2000, Unwind: jit.UnwindSkip, StackMaps: []jit.StackMap{{PCOffset: 16, FrameWords: 1, PointerMask: []byte{1}, HasUnwind: true, CallerPCOffset: 8, CallerSPOffset: 16}}}
	if err := jit.ValidateRegion(r); err != nil {
		t.Fatal(err)
	}
	bad := r
	bad.StackMaps = append([]jit.StackMap(nil), r.StackMaps...)
	bad.StackMaps[0].FrameOffset = 1
	var metadata *jit.MetadataError
	if !errors.As(jit.ValidateRegion(bad), &metadata) || metadata.Field != "FrameOffset" {
		t.Fatal("missing typed alignment diagnostic")
	}
	h := jit.Register(r)
	defer h.Unregister()
	r.StackMaps[0].PointerMask[0] = 0
	snapshot := h.Snapshot()
	if snapshot.StackMaps[0].PointerMask[0] != 1 {
		t.Fatal("caller changed published map")
	}
	snapshot.StackMaps[0].PointerMask[0] = 0
	if h.Snapshot().StackMaps[0].PointerMask[0] != 1 {
		t.Fatal("snapshot aliased published map")
	}
	func() {
		defer func() {
			if _, ok := recover().(*jit.MetadataError); !ok {
				t.Error("invalid append did not report typed metadata error")
			}
		}()
		h.AddStackMaps(jit.StackMap{PCOffset: 8, HasUnwind: true, CallerSPOffset: 16})
	}()
	if len(h.Snapshot().StackMaps) != 1 {
		t.Fatal("invalid append changed publication")
	}
	var out bytes.Buffer
	n, err := h.WriteTo(&out)
	if err != nil || n != int64(out.Len()) || !bytes.Contains(out.Bytes(), []byte("pc=0x1010")) {
		t.Fatalf("diagnostic: %s %v", out.String(), err)
	}
}

func TestEmissionExhaustionDoesNotCommitSequence(t *testing.T) {
	if !jit.SupportedEmitters() {
		t.Skip("ABI emitter target")
	}
	b := &jit.SliceBuffer{Bytes: make([]byte, 1)}
	defer func() {
		if _, ok := recover().(jit.CodeBufferFull); !ok {
			t.Error("missing typed emission overflow")
		}
		if b.Offset() != 0 || b.Bytes[0] != 0 {
			t.Error("partial ABI sequence")
		}
	}()
	jit.EmitFrame(b, jit.FramePlan{LocalBytes: 16})
}

func TestStackMapPaddingRemainsCompatible(t *testing.T) {
	// Historically only the first FrameWords bits are meaningful. Padding in
	// the final byte must not turn a valid existing registration into a panic.
	r := jit.Region{Start: 0x1000, End: 0x2000, StackMaps: []jit.StackMap{{FrameWords: 1, PointerMask: []byte{0xff}}}}
	if err := jit.ValidateRegion(r); err != nil {
		t.Fatal(err)
	}
}

func TestFrameRejectsEmitterScratch(t *testing.T) {
	if !jit.SupportedEmitters() {
		t.Skip("ABI emitter target")
	}
	b := &jit.SliceBuffer{Bytes: make([]byte, 512)}
	defer func() {
		if recover() == nil {
			t.Error("frame base aliases emitter scratch")
		}
		if b.Offset() != 0 {
			t.Error("invalid frame partially emitted")
		}
	}()
	jit.EmitFrame(b, jit.FramePlan{Frame: jit.RegisterConfig().Scratch})
}

func TestStackCheckRejectsUnreachableSlowPath(t *testing.T) {
	if !jit.SupportedEmitters() {
		t.Skip("ABI emitter target")
	}
	b := &jit.SliceBuffer{Bytes: make([]byte, 512)}
	defer func() {
		if recover() == nil {
			t.Error("out-of-range branch silently truncated")
		}
		if b.Offset() != 0 {
			t.Error("invalid branch partially emitted")
		}
	}()
	far := uint64(1) << 33
	jit.EmitStackCheck(b, 0, int(far))
}

func TestCallingConventionTextAndErrorResults(t *testing.T) {
	if !jit.SupportedEmitters() {
		t.Skip("native ABI helpers")
	}
	c := jit.GetCallingConventionForFunc((func() (string, error))(nil))
	if len(c.Outputs) != 2 || len(c.Outputs[0].Parts) != 2 || len(c.Outputs[1].Parts) != 2 {
		t.Fatalf("text/error result components: %+v", c.Outputs)
	}
	want := [][]bool{{true, false}, {false, true}}
	for i, value := range c.Outputs {
		for j, part := range value.Parts {
			if part.Pointer != want[i][j] {
				t.Fatalf("result %d part %d pointer=%v, want %v", i, j, part.Pointer, want[i][j])
			}
			if part.ValueOffset != uintptr(j)*reflect.TypeFor[uintptr]().Size() {
				t.Fatalf("result %d part %d has unexpected memory offset", i, j)
			}
		}
	}
}
