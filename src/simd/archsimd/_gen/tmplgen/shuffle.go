// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"simd/archsimd/_gen/sgutil"
	"strings"
	"text/template"
)

// emulatedShuffle describes an InterleaveLo, InterleaveHi, ConcatEven, or
// ConcatOdd method that amd64 emulates (see shuffles_emulated_amd64.go).
type emulatedShuffle struct {
	Op     string // "InterleaveLo", "InterleaveHi", "ConcatEven", "ConcatOdd"
	Base   string // "Int", "Uint", "Float"
	EWidth int    // 8, 16, 32, 64
	Vwidth int    // 128, 256, 512
	CPU    string // "AVX", "AVX2", "AVX512"
}

func (s emulatedShuffle) Count() int {
	return s.Vwidth / s.EWidth
}

func (s emulatedShuffle) VType() string {
	return fmt.Sprintf("%s%dx%d", s.Base, s.EWidth, s.Count())
}

func (s emulatedShuffle) UType() string {
	return fmt.Sprintf("Uint%dx%d", s.EWidth, s.Count())
}

func (s emulatedShuffle) LowerOp() string {
	return strings.ToLower(s.Op[:1]) + s.Op[1:]
}

func (s emulatedShuffle) Select2() string {
	if s.Op == "InterleaveHi" || s.Op == "ConcatOdd" {
		return "1, 3"
	}
	return "0, 2"
}

func (s emulatedShuffle) Select4() string {
	if s.Op == "ConcatOdd" {
		return "1, 3, 5, 7"
	}
	return "0, 2, 4, 6"
}

func (s emulatedShuffle) InterleaveOp() string {
	if s.Op == "ConcatOdd" {
		return "InterleaveHi"
	}
	return "InterleaveLo"
}

func (s emulatedShuffle) InterleaveOpGrouped() string {
	if s.Op == "ConcatOdd" {
		return "interleaveHiGrouped"
	}
	return "interleaveLoGrouped"
}

var (
	shuffleDelegateToUintTmpl = sgutil.TemplateNamed("shuffleDelegateToUintTmpl", `
// Emulated, CPU Feature: {{.CPU}}
func (x {{.VType}}) {{.Op}}(y {{.VType}}) (z {{.VType}}) {
	return x.ToBits().{{.Op}}(y.ToBits()).BitsTo{{.Base}}{{.EWidth}}()
}
`)

	shufflePermuteScalars2Tmpl = sgutil.TemplateNamed("shufflePermuteScalars2Tmpl", `
// Emulated, CPU Feature: {{.CPU}}
func (x {{.VType}}) {{.Op}}(y {{.VType}}) (z {{.VType}}) {
	return x.concatPermuteScalars({{.Select2}}, y)
}
`)

	shuffleConcatLookup512Tmpl = sgutil.TemplateNamed("shuffleConcatLookup512Tmpl", `
// Emulated, CPU Feature: {{.CPU}}
func (x {{.VType}}) {{.Op}}(y {{.VType}}) (z {{.VType}}) {
	return x.ConcatLookupMod(y, Load{{.UType}}Array(&{{.LowerOp}}{{.EWidth}}x{{.Count}}Idx))
}
`)

	interleave256IntTmpl = sgutil.TemplateNamed("interleave256IntTmpl", `
// Emulated, CPU Feature: {{.CPU}}
func (x {{.VType}}) {{.Op}}(y {{.VType}}) (z {{.VType}}) {
	lo := x.interleaveLoGrouped(y)
	hi := x.interleaveHiGrouped(y)
	return lo.concatPermute128Scalars({{.Select2}}, hi)
}
`)

	interleave256Float64Tmpl = sgutil.TemplateNamed("interleave256Float64Tmpl", `
// Emulated, CPU Feature: {{.CPU}}
func (x {{.VType}}) {{.Op}}(y {{.VType}}) (z {{.VType}}) {
	lo := x.concatPermuteScalarsGrouped(0, 2, y)
	hi := x.concatPermuteScalarsGrouped(1, 3, y)
	return lo.concatPermute128Scalars({{.Select2}}, hi)
}
`)

	interleave512Uint8Tmpl = sgutil.TemplateNamed("interleave512Uint8Tmpl", `
// Emulated, CPU Feature: {{.CPU}}
func (x Uint8x64) {{.Op}}(y Uint8x64) (z Uint8x64) {
	if X86.AVX512VBMI() {
		return x.ConcatLookupMod(y, LoadUint8x64Array(&{{.LowerOp}}8x64Idx))
	}
	lo := x.interleaveLoGrouped(y).ReshapeToUint64s()
	hi := x.interleaveHiGrouped(y).ReshapeToUint64s()
	return lo.ConcatLookupMod(hi, LoadUint64x8Array(&{{.LowerOp}}128x4Idx)).ReshapeToUint8s()
}
`)

	concat128Int64Tmpl = sgutil.TemplateNamed("concat128Int64Tmpl", `
// Emulated, CPU Feature: {{.CPU}}
func (x {{.VType}}) {{.Op}}(y {{.VType}}) (z {{.VType}}) {
	return x.{{.InterleaveOp}}(y)
}
`)

	concat128Elem32Tmpl = sgutil.TemplateNamed("concat128Elem32Tmpl", `
// Emulated, CPU Feature: {{.CPU}}
func (x {{.VType}}) {{.Op}}(y {{.VType}}) (z {{.VType}}) {
	return x.concatPermuteScalars({{.Select4}}, y)
}
`)

	concatEven128Uint16Tmpl = sgutil.TemplateNamed("concatEven128Uint16Tmpl", `
// Emulated, CPU Feature: {{.CPU}}
func (x Uint16x8) ConcatEven(y Uint16x8) (z Uint16x8) {
	idx := LoadInt8x16Array(&trunc32To16Idx)
	a := x.ReshapeToUint8s().LookupModOrZero(idx).ReshapeToUint64s()
	b := y.ReshapeToUint8s().LookupModOrZero(idx).ReshapeToUint64s()
	return a.InterleaveLo(b).ReshapeToUint16s()
}
`)

	concatOdd128Uint16Tmpl = sgutil.TemplateNamed("concatOdd128Uint16Tmpl", `
// Emulated, CPU Feature: {{.CPU}}
func (x Uint16x8) ConcatOdd(y Uint16x8) (z Uint16x8) {
	xo := x.ReshapeToUint32s().BitsToInt32().ShiftAllRight(16)
	yo := y.ReshapeToUint32s().BitsToInt32().ShiftAllRight(16)
	return xo.ConcatSaturateToInt16(yo).ToBits()
}
`)

	concatEven128Uint8Tmpl = sgutil.TemplateNamed("concatEven128Uint8Tmpl", `
// Emulated, CPU Feature: {{.CPU}}
func (x Uint8x16) ConcatEven(y Uint8x16) (z Uint8x16) {
	idx := LoadInt8x16Array(&trunc16To8Idx)
	a := x.LookupModOrZero(idx).ReshapeToUint64s()
	b := y.LookupModOrZero(idx).ReshapeToUint64s()
	return a.InterleaveLo(b).ReshapeToUint8s()
}
`)

	concatOdd128Uint8Tmpl = sgutil.TemplateNamed("concatOdd128Uint8Tmpl", `
// Emulated, CPU Feature: {{.CPU}}
func (x Uint8x16) ConcatOdd(y Uint8x16) (z Uint8x16) {
	xo := x.ReshapeToUint16s().BitsToInt16().ShiftAllRight(8)
	yo := y.ReshapeToUint16s().BitsToInt16().ShiftAllRight(8)
	return xo.concatSaturateToInt8(yo).ToBits()
}
`)

	concat256Float64Tmpl = sgutil.TemplateNamed("concat256Float64Tmpl", `
// Emulated, CPU Feature: {{.CPU}}
func (x Float64x4) {{.Op}}(y Float64x4) (z Float64x4) {
	lo := x.concatPermute128Scalars(0, 2, y)
	hi := x.concatPermute128Scalars(1, 3, y)
	return lo.concatPermuteScalarsGrouped({{.Select2}}, hi)
}
`)

	concat256Float32Tmpl = sgutil.TemplateNamed("concat256Float32Tmpl", `
// Emulated, CPU Feature: {{.CPU}}
func (x Float32x8) {{.Op}}(y Float32x8) (z Float32x8) {
	lo := x.concatPermute128Scalars(0, 2, y)
	hi := x.concatPermute128Scalars(1, 3, y)
	return lo.concatPermuteScalarsGrouped({{.Select4}}, hi)
}
`)

	concat256Uint64Tmpl = sgutil.TemplateNamed("concat256Uint64Tmpl", `
// Emulated, CPU Feature: {{.CPU}}
func (x Uint64x4) {{.Op}}(y Uint64x4) (z Uint64x4) {
	return gatherLo64(x.{{.InterleaveOpGrouped}}(y))
}
`)

	concat256Uint32Tmpl = sgutil.TemplateNamed("concat256Uint32Tmpl", `
// Emulated, CPU Feature: {{.CPU}}
func (x Uint32x8) {{.Op}}(y Uint32x8) (z Uint32x8) {
	return x.concatPermuteScalarsGrouped({{.Select4}}, y).LookupMod(LoadUint32x8Array(&gatherLo64Idx))
}
`)

	concatEven256Uint16Tmpl = sgutil.TemplateNamed("concatEven256Uint16Tmpl", `
// Emulated, CPU Feature: {{.CPU}}
func (x Uint16x16) ConcatEven(y Uint16x16) (z Uint16x16) {
	if X86.AVX512() {
		return x.ConcatLookupMod(y, LoadUint16x16Array(&concatEven16x16Idx))
	}
	idx := LoadInt8x32Array(&trunc32To16GroupedIdx)
	a := x.ReshapeToUint8s().LookupModOrZeroGrouped(idx).ReshapeToUint64s()
	b := y.ReshapeToUint8s().LookupModOrZeroGrouped(idx).ReshapeToUint64s()
	return gatherLo64(a.interleaveLoGrouped(b)).ReshapeToUint16s()
}
`)

	concatOdd256Uint16Tmpl = sgutil.TemplateNamed("concatOdd256Uint16Tmpl", `
// Emulated, CPU Feature: {{.CPU}}
func (x Uint16x16) ConcatOdd(y Uint16x16) (z Uint16x16) {
	if X86.AVX512() {
		return x.ConcatLookupMod(y, LoadUint16x16Array(&concatOdd16x16Idx))
	}
	xo := x.ReshapeToUint32s().BitsToInt32().ShiftAllRight(16)
	yo := y.ReshapeToUint32s().BitsToInt32().ShiftAllRight(16)
	return gatherLo64(xo.ConcatSaturateToInt16Grouped(yo).ToBits().ReshapeToUint64s()).ReshapeToUint16s()
}
`)

	concatEven256Uint8Tmpl = sgutil.TemplateNamed("concatEven256Uint8Tmpl", `
// Emulated, CPU Feature: {{.CPU}}
func (x Uint8x32) ConcatEven(y Uint8x32) (z Uint8x32) {
	if X86.AVX512VBMI() {
		return x.ConcatLookupMod(y, LoadUint8x32Array(&concatEven8x32Idx))
	}
	idx := LoadInt8x32Array(&trunc16To8GroupedIdx)
	a := x.LookupModOrZeroGrouped(idx).ReshapeToUint64s()
	b := y.LookupModOrZeroGrouped(idx).ReshapeToUint64s()
	return gatherLo64(a.interleaveLoGrouped(b)).ReshapeToUint8s()
}
`)

	concatOdd256Uint8Tmpl = sgutil.TemplateNamed("concatOdd256Uint8Tmpl", `
// Emulated, CPU Feature: {{.CPU}}
func (x Uint8x32) ConcatOdd(y Uint8x32) (z Uint8x32) {
	if X86.AVX512VBMI() {
		return x.ConcatLookupMod(y, LoadUint8x32Array(&concatOdd8x32Idx))
	}
	xo := x.ReshapeToUint16s().BitsToInt16().ShiftAllRight(8)
	yo := y.ReshapeToUint16s().BitsToInt16().ShiftAllRight(8)
	return gatherLo64(xo.concatSaturateToInt8Grouped(yo).ToBits().ReshapeToUint64s()).ReshapeToUint8s()
}
`)

	concat512Uint8Tmpl = sgutil.TemplateNamed("concat512Uint8Tmpl", `
// Emulated, CPU Feature: {{.CPU}}
func (x Uint8x64) {{.Op}}(y Uint8x64) (z Uint8x64) {
	if X86.AVX512VBMI() {
		return x.ConcatLookupMod(y, LoadUint8x64Array(&{{.LowerOp}}8x64Idx))
	}
	idx := LoadInt8x64Array(&{{.LowerOp}}8x64GroupedIdx)
	a := x.LookupModOrZeroGrouped(idx).ReshapeToUint64s()
	b := y.LookupModOrZeroGrouped(idx).ReshapeToUint64s()
	return a.ConcatLookupMod(b, LoadUint64x8Array(&concatEven64x8Idx)).ReshapeToUint8s()
}
`)
)

func interleaveTemplateFor(s emulatedShuffle) *template.Template {
	switch s.Vwidth {
	case 128:
		if s.EWidth == 64 {
			return shufflePermuteScalars2Tmpl
		}
		return shuffleDelegateToUintTmpl
	case 256:
		switch {
		case s.Base == "Float" && s.EWidth == 64:
			return interleave256Float64Tmpl
		case s.Base == "Float":
			return shuffleDelegateToUintTmpl
		default:
			return interleave256IntTmpl
		}
	case 512:
		if s.EWidth == 8 {
			if s.Base == "Int" {
				return shuffleDelegateToUintTmpl
			}
			return interleave512Uint8Tmpl
		}
		return shuffleConcatLookup512Tmpl
	}
	panic(fmt.Sprintf("unexpected interleave shape: %+v", s))
}

func concatTemplateFor(s emulatedShuffle) *template.Template {
	odd := s.Op == "ConcatOdd"
	switch s.Vwidth {
	case 128:
		switch s.EWidth {
		case 64:
			if s.Base == "Float" {
				return shufflePermuteScalars2Tmpl
			}
			return concat128Int64Tmpl
		case 32:
			return concat128Elem32Tmpl
		case 16:
			if s.Base == "Int" {
				return shuffleDelegateToUintTmpl
			}
			if odd {
				return concatOdd128Uint16Tmpl
			}
			return concatEven128Uint16Tmpl
		case 8:
			if s.Base == "Int" {
				return shuffleDelegateToUintTmpl
			}
			if odd {
				return concatOdd128Uint8Tmpl
			}
			return concatEven128Uint8Tmpl
		}
	case 256:
		if s.Base == "Float" {
			if s.EWidth == 64 {
				return concat256Float64Tmpl
			}
			return concat256Float32Tmpl
		}
		if s.Base == "Int" {
			return shuffleDelegateToUintTmpl
		}
		switch s.EWidth {
		case 64:
			return concat256Uint64Tmpl
		case 32:
			return concat256Uint32Tmpl
		case 16:
			if odd {
				return concatOdd256Uint16Tmpl
			}
			return concatEven256Uint16Tmpl
		case 8:
			if odd {
				return concatOdd256Uint8Tmpl
			}
			return concatEven256Uint8Tmpl
		}
	case 512:
		if s.EWidth == 8 {
			if s.Base == "Int" {
				return shuffleDelegateToUintTmpl
			}
			return concat512Uint8Tmpl
		}
		return shuffleConcatLookup512Tmpl
	}
	panic(fmt.Sprintf("unexpected concat shape: %+v", s))
}

// emulatedShuffles writes filename, which contains the amd64 InterleaveLo,
// InterleaveHi, ConcatEven, and ConcatOdd methods that are emulated using other
// operations.
func emulatedShuffles(filename string) {
	buf := files.NewGoFile(filename)
	prologue("tmplgen", "amd64", buf)
	exec := func(t *template.Template, s emulatedShuffle) {
		if err := t.Execute(buf, s); err != nil {
			panic(fmt.Errorf("template execute failed, %v", err))
		}
	}

	types := []struct {
		base string
		w    int
	}{
		{"Int", 8}, {"Uint", 8},
		{"Int", 16}, {"Uint", 16},
		{"Int", 32}, {"Uint", 32}, {"Float", 32},
		{"Int", 64}, {"Uint", 64}, {"Float", 64},
	}

	// InterleaveLo and InterleaveHi: 128-bit integers are native in ops_amd64.go.
	for _, op := range []string{"InterleaveLo", "InterleaveHi"} {
		for _, vw := range []int{128, 256, 512} {
			for _, typ := range types {
				if vw == 128 && typ.base != "Float" {
					continue
				}
				cpu := "AVX512"
				if vw == 128 || (vw == 256 && typ.base == "Float" && typ.w == 64) {
					cpu = "AVX"
				} else if vw == 256 {
					cpu = "AVX2"
				}
				s := emulatedShuffle{op, typ.base, typ.w, vw, cpu}
				exec(interleaveTemplateFor(s), s)
			}
		}
	}

	// ConcatEven and ConcatOdd: all widths and element types.
	for _, op := range []string{"ConcatEven", "ConcatOdd"} {
		for _, vw := range []int{128, 256, 512} {
			for _, typ := range types {
				cpu := "AVX512"
				if vw == 128 || (vw == 256 && typ.base == "Float") {
					cpu = "AVX"
				} else if vw == 256 {
					cpu = "AVX2"
				}
				s := emulatedShuffle{op, typ.base, typ.w, vw, cpu}
				exec(concatTemplateFor(s), s)
			}
		}
	}
}
