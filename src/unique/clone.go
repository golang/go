// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unique

import (
	"internal/abi"
	"unsafe"
)

// clone makes a copy of value, and may update string values found in value
// with a cloned version of those strings. The purpose of explicitly cloning
// strings is to avoid accidentally giving a large string a long lifetime.
//
// Note that this will clone strings in structs and arrays found in value,
// and will clone value if it itself is a string. It will not, however, clone
// strings if value is of interface or slice type (that is, found via an
// indirection).
func clone[T comparable](value T, seq *cloneSeq) T {
	for _, offset := range seq.stringOffsets {
		ps := (*string)(unsafe.Pointer(uintptr(unsafe.Pointer(&value)) + offset))
		*ps = cloneString(*ps)
	}
	return value
}

// singleStringClone describes how to clone a single string.
var singleStringClone = cloneSeq{stringOffsets: []uintptr{0}}

// cloneSeq describes how to clone a value of a particular type.
type cloneSeq struct {
	stringOffsets []uintptr
}

// makeCloneSeq creates a cloneSeq for a type.
func makeCloneSeq(typ *abi.Type) cloneSeq {
	if typ == nil {
		return cloneSeq{}
	}
	if typ.Kind() == abi.String {
		return singleStringClone
	}
	var seq cloneSeq
	switch typ.Kind() {
	case abi.Struct:
		buildStructCloneSeq(typ, &seq, 0)
	case abi.Array:
		buildArrayCloneSeq(typ, &seq, 0)
	}
	return seq
}

// buildStructCloneSeq populates a cloneSeq for an abi.Type that has Kind abi.Struct.
func buildStructCloneSeq(typ *abi.Type, seq *cloneSeq, baseOffset uintptr) {
	styp := typ.StructType()
	for i := range styp.Fields {
		f := &styp.Fields[i]
		switch f.Typ.Kind() {
		case abi.String:
			seq.stringOffsets = append(seq.stringOffsets, baseOffset+f.Offset)
		case abi.Struct:
			buildStructCloneSeq(f.Typ, seq, baseOffset+f.Offset)
		case abi.Array:
			buildArrayCloneSeq(f.Typ, seq, baseOffset+f.Offset)
		}
	}
}

// buildArrayCloneSeq populates a cloneSeq for an abi.Type that has Kind abi.Array.
func buildArrayCloneSeq(typ *abi.Type, seq *cloneSeq, baseOffset uintptr) {
	atyp := typ.ArrayType()
	etyp := atyp.Elem
	offset := baseOffset
	for range atyp.Len {
		switch etyp.Kind() {
		case abi.String:
			seq.stringOffsets = append(seq.stringOffsets, offset)
		case abi.Struct:
			buildStructCloneSeq(etyp, seq, offset)
		case abi.Array:
			buildArrayCloneSeq(etyp, seq, offset)
		}
		offset += etyp.Size()
		align := uintptr(etyp.FieldAlign())
		offset = (offset + align - 1) &^ (align - 1)
	}
}

// cloneString is a copy of strings.Clone, because we can't depend on the strings
// package here. Several packages that might make use of unique, like net, explicitly
// forbid depending on unicode, which strings depends on.
func cloneString(s string) string {
	if len(s) == 0 {
		return ""
	}
	b := make([]byte, len(s))
	copy(b, s)
	return unsafe.String(&b[0], len(b))
}
