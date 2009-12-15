// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ogle

import (
	"debug/proc"
	"math"
)

type Arch interface {
	// ToWord converts an array of up to 8 bytes in memory order
	// to a word.
	ToWord(data []byte) proc.Word
	// FromWord converts a word to an array of up to 8 bytes in
	// memory order.
	FromWord(v proc.Word, out []byte)
	// ToFloat32 converts a word to a float.  The order of this
	// word will be the order returned by ToWord on the memory
	// representation of a float, and thus may require reversing.
	ToFloat32(bits uint32) float32
	// FromFloat32 converts a float to a word.  This should return
	// a word that can be passed to FromWord to get the memory
	// representation of a float on this architecture.
	FromFloat32(f float32) uint32
	// ToFloat64 is to float64 as ToFloat32 is to float32.
	ToFloat64(bits uint64) float64
	// FromFloat64 is to float64 as FromFloat32 is to float32.
	FromFloat64(f float64) uint64

	// IntSize returns the number of bytes in an 'int'.
	IntSize() int
	// PtrSize returns the number of bytes in a 'uintptr'.
	PtrSize() int
	// FloatSize returns the number of bytes in a 'float'.
	FloatSize() int
	// Align rounds offset up to the appropriate offset for a
	// basic type with the given width.
	Align(offset, width int) int

	// G returns the current G pointer.
	G(regs proc.Regs) proc.Word

	// ClosureSize returns the number of bytes expected by
	// ParseClosure.
	ClosureSize() int
	// ParseClosure takes ClosureSize bytes read from a return PC
	// in a remote process, determines if the code is a closure,
	// and returns the frame size of the closure if it is.
	ParseClosure(data []byte) (frame int, ok bool)
}

type ArchLSB struct{}

func (ArchLSB) ToWord(data []byte) proc.Word {
	var v proc.Word
	for i, b := range data {
		v |= proc.Word(b) << (uint(i) * 8)
	}
	return v
}

func (ArchLSB) FromWord(v proc.Word, out []byte) {
	for i := range out {
		out[i] = byte(v)
		v >>= 8
	}
}

func (ArchLSB) ToFloat32(bits uint32) float32 {
	// TODO(austin) Do these definitions depend on my current
	// architecture?
	return math.Float32frombits(bits)
}

func (ArchLSB) FromFloat32(f float32) uint32 { return math.Float32bits(f) }

func (ArchLSB) ToFloat64(bits uint64) float64 { return math.Float64frombits(bits) }

func (ArchLSB) FromFloat64(f float64) uint64 { return math.Float64bits(f) }

type ArchAlignedMultiple struct{}

func (ArchAlignedMultiple) Align(offset, width int) int {
	return ((offset - 1) | (width - 1)) + 1
}

type amd64 struct {
	ArchLSB
	ArchAlignedMultiple
	gReg int
}

func (a *amd64) IntSize() int { return 4 }

func (a *amd64) PtrSize() int { return 8 }

func (a *amd64) FloatSize() int { return 4 }

func (a *amd64) G(regs proc.Regs) proc.Word {
	// See src/pkg/runtime/mkasmh
	if a.gReg == -1 {
		ns := regs.Names()
		for i, n := range ns {
			if n == "r15" {
				a.gReg = i
				break
			}
		}
	}

	return regs.Get(a.gReg)
}

func (a *amd64) ClosureSize() int { return 8 }

func (a *amd64) ParseClosure(data []byte) (int, bool) {
	if data[0] == 0x48 && data[1] == 0x81 && data[2] == 0xc4 && data[7] == 0xc3 {
		return int(a.ToWord(data[3:7]) + 8), true
	}
	return 0, false
}

var Amd64 = &amd64{gReg: -1}
