// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"simd/archsimd/_gen/sgutil"
	"text/template"
)

// narrowConversion is a narrowing integer conversion that amd64 emulates (see
// converts_emulated_amd64.go).
type narrowConversion struct {
	Base    string // base type of the input, "Int" or "Uint"
	EWidth  int    // element width of the input
	Vwidth  int    // width of the input and result vectors
	OBase   string // base type of the result
	OEWidth int    // element width of the result
	CPU     string // CPU feature the emulation requires
}

// VType returns the input vector type, e.g., Int16x8.
func (c narrowConversion) VType() string {
	return fmt.Sprintf("%s%dx%d", c.Base, c.EWidth, c.Count())
}

// Count returns the number of elements of the input.
func (c narrowConversion) Count() int {
	return c.Vwidth / c.EWidth
}

// OVType returns the result vector type, e.g., Int8x16.
func (c narrowConversion) OVType() string {
	return fmt.Sprintf("%s%dx%d", c.OBase, c.OEWidth, c.Vwidth/c.OEWidth)
}

// Pack returns the name of the operation that packs two vectors of type
// c.VType into one vector of type c.OVType with saturation. VPACKSSWB and
// VPACKUSWB are internal, since there is no exported concatenating conversion
// to 8-bit elements.
func (c narrowConversion) Pack() string {
	if c.OEWidth == 8 {
		return fmt.Sprintf("concatSaturateTo%s8", c.OBase)
	}
	return fmt.Sprintf("ConcatSaturateTo%s%d", c.OBase, c.OEWidth)
}

// truncTemplate emulates the 2:1 TruncTo conversions on 128- and 256-bit
// vectors. The signed forms use the unsigned forms. The unsigned forms
// shuffle the low part of each element into place, within each 128-bit group
// at 256 bits.
var truncTemplate = sgutil.TemplateNamed("truncTemplate", `
// Emulated, CPU Feature: {{.CPU}}
func (x {{.VType}}) TruncTo{{.OBase}}{{.OEWidth}}() (z {{.OVType}}) {
{{- if eq .Base "Int"}}
	return x.ToBits().TruncToUint{{.OEWidth}}().BitsToInt{{.OEWidth}}()
{{- else if eq .Vwidth 128}}
{{-  if eq .EWidth 64}}
	var zero Uint32x4
	return x.ReshapeToUint32s().ConcatPermuteScalars(0, 2, 4, 4, zero)
{{-  else}}
	return x.ReshapeToUint8s().LookupModOrZero(LoadInt8x16Array(&trunc{{.EWidth}}To{{.OEWidth}}Idx)){{if ne .OEWidth 8}}.ReshapeToUint{{.OEWidth}}s(){{end}}
{{-  end}}
{{- else}}
	if X86.AVX512() {
		return x.truncToUint{{.OEWidth}}()
	}
{{-  if eq .EWidth 64}}
	var zero Uint32x8
	g := x.ReshapeToUint32s().ConcatPermuteScalarsGrouped(0, 2, 4, 4, zero)
{{-  else}}
	g := x.ReshapeToUint8s().LookupModOrZeroGrouped(LoadInt8x32Array(&trunc{{.EWidth}}To{{.OEWidth}}GroupedIdx))
{{-  end}}
	return gatherLo64(g.ReshapeToUint64s()).ReshapeToUint{{.OEWidth}}s()
{{- end}}
}
`)

// saturateTemplate emulates the 2:1 SaturateTo conversions on 128- and 256-bit
// vectors.
var saturateTemplate = sgutil.TemplateNamed("saturateTemplate", `
// Emulated, CPU Feature: {{.CPU}}
func (x {{.VType}}) SaturateTo{{.OBase}}{{.OEWidth}}() (z {{.OVType}}) {
{{- if eq .Base "Uint"}}
	if X86.AVX512() {
		return x.saturateToUint{{.OEWidth}}()
	}
{{-  if eq .EWidth 64}}
	// x fits in a uint32 if its high 32 bits are zero.
	var zero {{.VType}}
	hi := LoadInt64x{{.Count}}Array(&maxUint32x{{.Count}}).ToBits()
	return x.IfElse(x.ShiftAllRight(32).Equal(zero), hi).TruncToUint32()
{{-  else}}
	// After the Min, x is non-negative as an int{{.EWidth}}, so the signed-to-unsigned
	// saturation doesn't change it.
	x = x.Min(Load{{.VType}}Array(&maxUint{{.OEWidth}}x{{.Count}}))
	return x.BitsToInt{{.EWidth}}().SaturateToUint{{.OEWidth}}()
{{-  end}}
{{- else if and (eq .EWidth 64) (eq .OBase "Int")}}
	if X86.AVX512() {
		return x.saturateToInt32()
	}
	hi, lo := Load{{.VType}}Array(&maxInt32x{{.Count}}), Load{{.VType}}Array(&minInt32x{{.Count}})
	x = hi.IfElse(x.Greater(hi), x)
	x = lo.IfElse(x.Less(lo), x)
	return x.TruncToInt32()
{{- else if eq .EWidth 64}}
	var zero {{.VType}}
	if X86.AVX512() {
		return x.Max(zero).ToBits().saturateToUint32()
	}
	hi := Load{{.VType}}Array(&maxUint32x{{.Count}})
	x = x.Masked(x.Greater(zero))
	x = hi.IfElse(x.Greater(hi), x)
	return x.ToBits().TruncToUint32()
{{- else if eq .Vwidth 128}}
	var zero {{.VType}}
	return x.{{.Pack}}(zero)
{{- else if eq .OBase "Int"}}
	if X86.AVX512() {
		return x.saturateToInt{{.OEWidth}}()
	}
	var zero {{.VType}}
	g := x.{{.Pack}}Grouped(zero)
	return gatherLo64(g.ToBits().ReshapeToUint64s()).ReshapeToUint{{.OEWidth}}s().BitsToInt{{.OEWidth}}()
{{- else}}
	var zero {{.VType}}
	g := x.{{.Pack}}Grouped(zero)
	return gatherLo64(g.ReshapeToUint64s()).ReshapeToUint{{.OEWidth}}s()
{{- end}}
}
`)

// saturateSignedToUnsignedTemplate implements the SaturateToUint conversions
// from signed integers that have no AVX-512 instruction and aren't emulated by
// saturateTemplate. These clamp negative values to 0 and then use unsigned
// saturation.
var saturateSignedToUnsignedTemplate = sgutil.TemplateNamed("saturateSignedToUnsignedTemplate", `
// Emulated, CPU Feature: {{.CPU}}
func (x {{.VType}}) SaturateToUint{{.OEWidth}}() (z {{.OVType}}) {
	var zero {{.VType}}
	return x.Max(zero).ToBits().SaturateToUint{{.OEWidth}}()
}
`)

// narrowConversions writes filename, which contains the amd64 narrowing
// integer conversions that are emulated using other operations.
func narrowConversions(filename string) {
	buf := files.NewGoFile(filename)
	prologue("tmplgen", "amd64", buf)
	exec := func(t *template.Template, c narrowConversion) {
		if err := t.Execute(buf, c); err != nil {
			panic(fmt.Errorf("template execute failed, %v", err))
		}
	}
	cpu := map[int]string{128: "AVX", 256: "AVX2"}
	for _, vw := range []int{128, 256} {
		for _, w := range []int{16, 32, 64} {
			for _, base := range []string{"Int", "Uint"} {
				exec(truncTemplate, narrowConversion{base, w, vw, base, w / 2, cpu[vw]})
			}
		}
	}
	for _, vw := range []int{128, 256} {
		for _, w := range []int{16, 32, 64} {
			exec(saturateTemplate, narrowConversion{"Int", w, vw, "Int", w / 2, cpu[vw]})
			exec(saturateTemplate, narrowConversion{"Int", w, vw, "Uint", w / 2, cpu[vw]})
			exec(saturateTemplate, narrowConversion{"Uint", w, vw, "Uint", w / 2, cpu[vw]})
		}
	}
	for _, ow := range []int{8, 16, 32} {
		for _, w := range []int{16, 32, 64} {
			for _, vw := range []int{128, 256, 512} {
				if w <= ow || (w == 2*ow && vw < 512) {
					// Same or wider result, or emulated by saturateTemplate.
					continue
				}
				exec(saturateSignedToUnsignedTemplate, narrowConversion{"Int", w, vw, "Uint", ow, "AVX512"})
			}
		}
	}
}
