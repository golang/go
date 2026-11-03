// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit

package jit

import "reflect"

// MemoryPart describes a scalar component of an in-memory Go value. Padding
// bytes are omitted. Pointer identifies a GC pointer, rather than an integer
// which happens to contain an address. For interfaces only the data word is a
// GC pointer; the type/itab word is runtime metadata. This describes the current
// toolchain representation, not a persisted or cross-version storage format.
type MemoryPart struct {
	Offset, Size uintptr
	Pointer      bool
}

// MemoryLayout describes the representation of a Go value independently of
// argument registers, spill homes and call frames.
type MemoryLayout struct {
	Type        reflect.Type
	Size, Align uintptr
	Parts       []MemoryPart
}

// GetMemoryLayout returns scalar components at their actual memory offsets,
// including fields of aggregates assigned wholly to the stack by the call ABI.
// It panics for a nil type. Returned slices belong to the caller.
func GetMemoryLayout(t reflect.Type) MemoryLayout {
	if t == nil {
		panic("runtime/jit: expected memory type")
	}
	m := MemoryLayout{Type: t, Size: t.Size(), Align: uintptr(t.Align())}
	for _, p := range stackParts(t, 0, 0) {
		m.Parts = append(m.Parts, MemoryPart{p.ValueOffset, p.Size, p.Pointer})
	}
	return m
}

// Store writes a typed Go value with the running toolchain's write barriers.
// Call generated specializations through an ordinary Go-call boundary, with
// live roots spilled and a safepoint map. A pointer-free new value alone never
// justifies omitting the barrier for a pointer-bearing destination: an old
// pointer being overwritten may still require shading.
func Store[T any](dst *T, value T) { *dst = value }

// Copy copies a typed value, including the required deletion/insertion barriers.
// Source and destination may overlap. The Go-call requirements of Store apply.
func Copy[T any](dst, src *T) { *dst = *src }

// Clear zeros a typed value, including barriers for overwritten pointers.
// The Go-call requirements of Store apply.
func Clear[T any](dst *T) { var zero T; *dst = zero }
