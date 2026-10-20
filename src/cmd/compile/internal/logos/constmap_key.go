// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

// This file gives the M5-S4 const map keys their compile-time view (SPEC
// T-Sec9): the base hash that places a key (constKeyHash, mirroring the
// hash expression the lowered lookup computes), the canonical bytes that
// break hash ties (constKeyCanon), and the Go order that range follows
// (constKeyCompare). Keys are evaluated ctfe values; a const map key of a
// type these functions cannot hash never reaches here, because the checker
// rejects it (types2 logosConstMapKeyOK).

import (
	"cmd/compile/internal/logos/ctfe"
	"math"
	"math/big"
	"strconv"
	"strings"
)

// constKeyMask is 2^64-1, the mask that turns a big integer into its
// 64-bit two's-complement bits.
var constKeyMask = new(big.Int).SetUint64(math.MaxUint64)

// constKeyIsInt reports whether k is an integer kind.
func constKeyIsInt(k ctfe.Kind) bool {
	switch k {
	case ctfe.KInt, ctfe.KInt8, ctfe.KInt16, ctfe.KInt32, ctfe.KInt64,
		ctfe.KUint, ctfe.KUint8, ctfe.KUint16, ctfe.KUint32, ctfe.KUint64, ctfe.KUintptr:
		return true
	}
	return false
}

// constKeyIsFloat reports whether k is a float kind.
func constKeyIsFloat(k ctfe.Kind) bool {
	return k == ctfe.KFloat32 || k == ctfe.KFloat64
}

// constKeyBits returns the 64-bit two's-complement bits of an integer key.
func constKeyBits(v ctfe.Value) uint64 {
	return new(big.Int).And(v.BigInt(), constKeyMask).Uint64()
}

// constKeyFloat returns a float key widened to float64. A float32 key is
// first rounded to float32, so both sides hash the same bits.
func constKeyFloat(v ctfe.Value) float64 {
	if v.Type().Kind == ctfe.KFloat32 {
		return float64(float32(v.Float64()))
	}
	return v.Float64()
}

// constKeyHash returns the base hash of a const map key. It mirrors the
// lowered hash (constKeyHashText): string, bool, integer, and float keys
// hash by value; struct and array keys fold their non-blank fields and
// elements in order with rt.HashCombine.
func constKeyHash(v ctfe.Value) uint64 {
	t := v.Type()
	switch {
	case t.Kind == ctfe.KString:
		return constHashString(v.Str())
	case t.Kind == ctfe.KBool:
		return constHashBool(v.Bool())
	case constKeyIsFloat(t.Kind):
		return constHashFloat64(constKeyFloat(v))
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

// constKeyCanon returns the canonical bytes of a key: the tie-break among
// keys with equal base hashes. Equal keys have equal canonical bytes, and
// struct fields and array elements are delimited so distinct keys differ.
func constKeyCanon(v ctfe.Value) string {
	var b strings.Builder
	constKeyCanonInto(&b, v)
	return b.String()
}

func constKeyCanonInto(b *strings.Builder, v ctfe.Value) {
	t := v.Type()
	switch {
	case t.Kind == ctfe.KString:
		b.WriteString(strconv.Itoa(len(v.Str())))
		b.WriteByte(':')
		b.WriteString(v.Str())
	case t.Kind == ctfe.KBool:
		if v.Bool() {
			b.WriteByte('t')
		} else {
			b.WriteByte('f')
		}
	case constKeyIsFloat(t.Kind):
		f := constKeyFloat(v)
		if f == 0 {
			f = 0 // negative zero keys as positive zero, as the hash does
		}
		b.WriteString(strconv.FormatUint(math.Float64bits(f), 16))
	case constKeyIsInt(t.Kind):
		b.WriteString(strconv.FormatUint(constKeyBits(v), 16))
	case t.Kind == ctfe.KStruct:
		b.WriteByte('{')
		for i, f := range t.Fields {
			if f.Name != "_" {
				constKeyCanonInto(b, v.Fields()[i])
				b.WriteByte(',')
			}
		}
		b.WriteByte('}')
	case t.Kind == ctfe.KArray:
		b.WriteByte('[')
		for _, e := range v.Elems() {
			constKeyCanonInto(b, e)
			b.WriteByte(',')
		}
		b.WriteByte(']')
	}
}

// constKeyCompare orders two keys of one const map the way Go orders them:
// strings bytewise, bools false before true, integers and floats
// numerically, and structs and arrays lexicographically over their
// non-blank fields and elements. Range visits keys in this order.
func constKeyCompare(a, b ctfe.Value) int {
	t := a.Type()
	switch {
	case t.Kind == ctfe.KString:
		return strings.Compare(a.Str(), b.Str())
	case t.Kind == ctfe.KBool:
		return constKeyCmpBool(a.Bool(), b.Bool())
	case constKeyIsFloat(t.Kind):
		fa, fb := constKeyFloat(a), constKeyFloat(b)
		switch {
		case fa < fb:
			return -1
		case fa > fb:
			return 1
		}
		return 0
	case constKeyIsInt(t.Kind):
		return a.BigInt().Cmp(b.BigInt())
	case t.Kind == ctfe.KStruct:
		for i, f := range t.Fields {
			if f.Name == "_" {
				continue
			}
			if c := constKeyCompare(a.Fields()[i], b.Fields()[i]); c != 0 {
				return c
			}
		}
		return 0
	case t.Kind == ctfe.KArray:
		ea, eb := a.Elems(), b.Elems()
		for i := range ea {
			if c := constKeyCompare(ea[i], eb[i]); c != 0 {
				return c
			}
		}
		return 0
	}
	return 0
}

func constKeyCmpBool(a, b bool) int {
	switch {
	case a == b:
		return 0
	case !a:
		return -1
	}
	return 1
}
