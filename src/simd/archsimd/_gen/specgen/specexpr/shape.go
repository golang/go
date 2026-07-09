// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specexpr

import (
	"fmt"
	"strings"
)

// MinWidth is the minimum vector width, in bits.
const MinWidth = Int(128)

type Type interface {
	isType()
	String() string
}

type Basic struct {
	Base string // "int", "uint", "float", "Mask", etc
	Bits Int    // 0 if unsized, otherwise >= 8
}

var MakeBasic = MakeFunc2("Basic", func(base string, bits Int) (any, error) {
	// Perform width rounding.
	bits = max(8, bits)
	return Basic{Base: base, Bits: bits}, nil
})

func (t Basic) isType() {}
func (t Basic) String() string {
	if t.Bits == 0 {
		return t.Base
	}
	return fmt.Sprintf("%s%d", t.Base, t.Bits)
}

type Vector struct {
	Elem  Basic // Must have Bits != 0
	Width Num   // Bit width
}

var MakeVector = MakeFunc2("VectorW", func(elem Basic, w Num) (any, error) {
	if !w.ValidWidth() {
		return nil, fmt.Errorf("invalid width %s", w)
	}
	return Vector{Elem: elem, Width: w}, nil
})

var makeVectorL = MakeFunc2("VectorL", func(elem Basic, l Num) (any, error) {
	w, _ := l.Mul(elem.Bits)
	if w2, ok := w.(Int); ok {
		// Perform width rounding
		w = max(MinWidth, w2)
	}
	if !w.ValidWidth() {
		return nil, fmt.Errorf("invalid width %s", w)
	}
	return Vector{Elem: elem, Width: w}, nil
})

func (t Vector) isType() {}
func (t Vector) String() string {
	var buf strings.Builder
	if t.Elem.Base == "" {
		buf.WriteString("<bad Elem>")
	} else {
		buf.WriteString(strings.ToTitle(t.Elem.Base[:1]))
		buf.WriteString(t.Elem.Base[1:])
		fmt.Fprintf(&buf, "%d", t.Elem.Bits)
	}
	if t.Scalable() {
		buf.WriteString("s")
	} else {
		l, err := t.Width.Div(t.Elem.Bits)
		if err == nil {
			fmt.Fprintf(&buf, "x%d", l)
		} else {
			// Bad width, but we can print it anyway
			fmt.Fprintf(&buf, "w%s", t.Width)
		}
	}
	return buf.String()
}
func (t Vector) Scalable() bool {
	sw, ok := t.Width.(ScalableWidth)
	return ok && sw.ValidWidth()
}

type Pointer struct {
	Elem Type
}

var MakePointer = MakeFunc1("Pointer", func(elem Type) (any, error) {
	return Pointer{elem}, nil
})

func (t Pointer) isType() {}
func (t Pointer) String() string {
	return "*" + t.Elem.String()
}

type Array struct {
	Elem Type
	Len  Int
}

var MakeArray = MakeFunc2("Array", func(elem Type, len Int) (any, error) {
	return Array{elem, len}, nil
})

func (t Array) isType() {}
func (t Array) String() string {
	return fmt.Sprintf("[%d]%s", t.Len, t.Elem)
}

type Slice struct {
	Elem Type
}

var MakeSlice = MakeFunc1("Slice", func(elem Type) (any, error) {
	return Slice{elem}, nil
})

func (t Slice) isType() {}
func (t Slice) String() string {
	return "[]" + t.Elem.String()
}
