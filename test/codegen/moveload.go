// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// From issue #77720: cmd/compile: field access on struct-returning method copies entire struct

type moveLoadBig struct {
	typ   int8
	index int64
	str   string
	pkgID string
}

type moveLoadHandle[T any] struct {
	value *T
}

func (h moveLoadHandle[T]) Value() T { return *h.value }

type moveLoadS struct {
	h moveLoadHandle[moveLoadBig]
}

func moveLoadFieldViaValue(s moveLoadS) int8 {
	// amd64:-`MOVUPS`
	// amd64:`MOVBLZX`
	return s.h.Value().typ
}

func moveLoadFieldViaValueInline(ss []moveLoadS, i int) int8 {
	// amd64:-`MOVUPS`
	// amd64:`MOVBLZX`
	return ss[i&7].h.Value().typ
}
