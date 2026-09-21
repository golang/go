// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package abi

import (
	"unsafe"
)

// Map constants common to several packages
// runtime/runtime-gdb.py:MapTypePrinter contains its own copy
const (
	// Number of bits in the group.slot count.
	MapGroupSlotsBits = 3

	// Number of slots in a group.
	MapGroupSlots = 1 << MapGroupSlotsBits // 8

	// Maximum key or elem size to keep inline (instead of mallocing per element).
	// Must fit in a uint8.
	MapMaxKeyBytes  = 128
	MapMaxElemBytes = 128

	ctrlEmpty = 0b10000000
	bitsetLSB = 0x0101010101010101

	// Value of control word with all empty slots.
	MapCtrlEmpty = bitsetLSB * uint64(ctrlEmpty)
)

type MapType struct {
	Type
	Key   *Type
	Elem  *Type
	Group *Type // internal type representing a slot group
	// function for hashing keys (ptr to key, seed) -> hash
	Hasher    func(unsafe.Pointer, uintptr) uintptr
	GroupSize uintptr // == Group.Size_
	// These fields describe how to access keys and elems within a group.
	// The formulas key(i) = KeysOff + i*KeyStride and
	// elem(i) = ElemsOff + i*ElemStride work for both group layouts:
	//
	// With GOEXPERIMENT=mapsplitgroup (split arrays KKKKVVVV):
	//   KeysOff    = offset of keys array in group
	//   KeyStride  = size of a single key
	//   ElemsOff   = offset of elems array in group
	//   ElemStride = size of a single elem
	//
	// Without (interleaved slots KVKVKVKV):
	//   KeysOff    = offset of slots array in group
	//   KeyStride  = size of a key/elem slot (stride between keys)
	//   ElemsOff   = offset of first elem (slots offset + elem offset within slot)
	//   ElemStride = size of a key/elem slot (stride between elems)
	KeysOff    uintptr
	KeyStride  uintptr
	ElemsOff   uintptr
	ElemStride uintptr
	ElemOff    uintptr // GOEXPERIMENT=nomapsplitgroup only
	Flags      uint32
}

// Flag values
const (
	MapNeedKeyUpdate = 1 << iota
	MapHashMightPanic
	MapIndirectKey
	MapIndirectElem
)

func (mt *MapType) NeedKeyUpdate() bool { // true if we need to update key on an overwrite
	return mt.Flags&MapNeedKeyUpdate != 0
}
func (mt *MapType) HashMightPanic() bool { // true if hash function might panic
	return mt.Flags&MapHashMightPanic != 0
}
func (mt *MapType) IndirectKey() bool { // store ptr to key instead of key itself
	return mt.Flags&MapIndirectKey != 0
}
func (mt *MapType) IndirectElem() bool { // store ptr to elem instead of elem itself
	return mt.Flags&MapIndirectElem != 0
}
