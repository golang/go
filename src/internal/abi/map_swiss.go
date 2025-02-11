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
	SwissMapGroupSlotsBits = 3

	// Number of slots in a group.
	SwissMapGroupSlots = 1 << SwissMapGroupSlotsBits // 8

	// Maximum key or elem size to keep inline (instead of mallocing per element).
	// Must fit in a uint8.
	SwissMapMaxKeyBytes  = 128
	SwissMapMaxElemBytes = 128

	ctrlEmpty = 0b10000000
	bitsetLSB = 0x0101010101010101

	// Value of control word with all empty slots.
	SwissMapCtrlEmpty = bitsetLSB * uint64(ctrlEmpty)
)

type SwissMapType struct {
	Type
	Elem  *Type
	Key   *Type
	Group *Type // internal type representing a slot group
	// function for hashing keys (ptr to key, seed) -> hash
	Hasher    func(unsafe.Pointer, uintptr) uintptr
	GroupSize uintptr // == Group.Size_
	SlotSize  uintptr // size of key/elem slot
	ElemOff   uintptr // offset of elem in key/elem slot
	Flags     uint32
}

// Flag values
const (
	SwissMapNeedKeyUpdate = 1 << iota
	SwissMapHashMightPanic
	SwissMapIndirectKey
	SwissMapIndirectElem
)

func (mt *SwissMapType) NeedKeyUpdate() bool { // true if we need to update key on an overwrite
	return mt.Flags&SwissMapNeedKeyUpdate != 0
}
func (mt *SwissMapType) HashMightPanic() bool { // true if hash function might panic
	return mt.Flags&SwissMapHashMightPanic != 0
}
func (mt *SwissMapType) IndirectKey() bool { // store ptr to key instead of key itself
	return mt.Flags&SwissMapIndirectKey != 0
}
func (mt *SwissMapType) IndirectElem() bool { // store ptr to elem instead of elem itself
	return mt.Flags&SwissMapIndirectElem != 0
}
