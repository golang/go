// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"cmd/compile/internal/logos/ctfe"
	"math"
	"math/big"
	"strconv"
	"strings"
)

var constKeyMask = new(big.Int).SetUint64(math.MaxUint64)

func constKeyIsInt(k ctfe.Kind) bool {
	switch k {
	case ctfe.KInt, ctfe.KInt8, ctfe.KInt16, ctfe.KInt32, ctfe.KInt64,
		ctfe.KUint, ctfe.KUint8, ctfe.KUint16, ctfe.KUint32, ctfe.KUint64, ctfe.KUintptr:
		return true
	}
	return false
}

func constKeyIsFloat(k ctfe.Kind) bool {
	return k == ctfe.KFloat32 || k == ctfe.KFloat64
}

func constKeyIsComplex(k ctfe.Kind) bool {
	return k == ctfe.KComplex64 || k == ctfe.KComplex128
}

func constKeyBits(v ctfe.Value) uint64 {
	return new(big.Int).And(v.BigInt(), constKeyMask).Uint64()
}

func constKeyFloat(v ctfe.Value) float64 {
	if v.Type().Kind == ctfe.KFloat32 {
		return float64(float32(v.Float64()))
	}
	return v.Float64()
}

func constKeyComplex(v ctfe.Value) complex128 {
	c := v.Complex()
	if v.Type().Kind == ctfe.KComplex64 {
		return complex(float64(float32(real(c))), float64(float32(imag(c))))
	}
	return c
}

func constKeyHash(v ctfe.Value) uint64 {
	t := v.Type()
	switch {
	case t.Kind == ctfe.KString:
		return constHashString(v.Str())
	case t.Kind == ctfe.KBool:
		return constHashBool(v.Bool())
	case constKeyIsFloat(t.Kind):
		return constHashFloat64(constKeyFloat(v))
	case constKeyIsComplex(t.Kind):
		c := constKeyComplex(v)
		return constHashCombine(constHashFloat64(real(c)), constHashFloat64(imag(c)))
	case constKeyIsInt(t.Kind):
		return constHashUint64(constKeyBits(v))
	case t.Kind == ctfe.KStruct:
		h := uint64(0)
		for i, f := range t.Fields {
			if f.Name != "_" {
				h = constHashCombine(h, constKeyHash(v.Fields()[i]))
			}
		}
		return h
	case t.Kind == ctfe.KArray:
		h := uint64(0)
		for _, e := range v.Elems() {
			h = constHashCombine(h, constKeyHash(e))
		}
		return h
	}
	panic("logos: const map key kind is not hashable: " + strconv.Itoa(int(t.Kind)))
}

func constKeyCanon(v ctfe.Value) string {
	t := v.Type()
	switch {
	case t.Kind == ctfe.KString:
		return strconv.Itoa(len(v.Str())) + ":" + v.Str()
	case t.Kind == ctfe.KBool:
		if v.Bool() {
			return "t"
		}
		return "f"
	case constKeyIsFloat(t.Kind):
		f := constKeyFloat(v)
		if f == 0 {
			f = 0
		}
		return strconv.FormatUint(math.Float64bits(f), 16)
	case constKeyIsComplex(t.Kind):
		c := constKeyComplex(v)
		re, im := real(c), imag(c)
		if re == 0 {
			re = 0
		}
		if im == 0 {
			im = 0
		}
		return strconv.FormatUint(math.Float64bits(re), 16) + "," + strconv.FormatUint(math.Float64bits(im), 16)
	case constKeyIsInt(t.Kind):
		return strconv.FormatUint(constKeyBits(v), 16)
	case t.Kind == ctfe.KStruct:
		b := strings.Builder{}
		b.WriteByte('{')
		for i, f := range t.Fields {
			if f.Name != "_" {
				b.WriteString(constKeyCanon(v.Fields()[i]))
				b.WriteByte(',')
			}
		}
		b.WriteByte('}')
		return b.String()
	case t.Kind == ctfe.KArray:
		b := strings.Builder{}
		b.WriteByte('[')
		for _, e := range v.Elems() {
			b.WriteString(constKeyCanon(e))
			b.WriteByte(',')
		}
		b.WriteByte(']')
		return b.String()
	}
	return ""
}
