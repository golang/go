// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

func (v Value) IsNilValue() bool { return v.isNil }

func (v Value) NoLiveReferences() bool {
	if !v.IsValid() {
		return false
	}
	switch v.typ.Kind {
	case KInvalid:
		return v.typ.Impure != "" && v.isNil
	case KSlice, KMap, KUntypedNil:
		return v.isNil
	case KStruct:
		if len(v.strct) != len(v.typ.Fields) {
			return false
		}
		for _, field := range v.strct {
			if !field.NoLiveReferences() {
				return false
			}
		}
		return true
	case KArray:
		if int64(len(v.array)) != v.typ.Len {
			return false
		}
		for _, elem := range v.array {
			if !elem.NoLiveReferences() {
				return false
			}
		}
		return true
	case KBool, KInt, KInt8, KInt16, KInt32, KInt64,
		KUint, KUint8, KUint16, KUint32, KUint64, KUintptr,
		KFloat32, KFloat64, KString, KUntypedInt, KUntypedFloat,
		KUntypedRune, KUntypedString, KUntypedBool, KComplex64,
		KComplex128, KUntypedComplex:
		return true
	}
	return false
}

func (v Value) SliceHeader() (backing []Value, lo, length, cap_ int, ok bool) {
	if v.slice == nil || v.slice.backing == nil {
		return nil, 0, 0, 0, false
	}
	return *v.slice.backing, v.slice.lo, v.slice.length, v.slice.cap_, true
}

func (v Value) SliceNeedsBacking() bool {
	if v.slice == nil {
		return false
	}
	return v.slice.lo != 0 || v.slice.cap_ != v.slice.length
}

func (v Value) sliceAccessible() []Value {
	if v.slice == nil || v.slice.backing == nil {
		return nil
	}
	b := *v.slice.backing
	lo, hi := v.slice.lo, v.slice.cap_
	if lo < 0 || lo > hi || hi > len(b) {
		return nil
	}
	return b[lo:hi]
}

func (v Value) sliceVisible() []Value {
	if v.slice == nil || v.slice.backing == nil {
		return nil
	}
	b := *v.slice.backing
	lo, hi := v.slice.lo, v.slice.lo+v.slice.length
	if lo < 0 || hi > len(b) || lo > hi {
		return nil
	}
	return b[lo:hi]
}
