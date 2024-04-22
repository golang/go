// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package abi is a temporary copy of the swissmap abi. It will be eliminated
// once swissmaps are integrated into the runtime.
package abi

import (
	"internal/abi"
	"unsafe"
)

// Map constants common to several packages
// runtime/runtime-gdb.py:MapTypePrinter contains its own copy
const (
	// Number of slots in a group.
	SwissMapGroupSlots = 8
)

type SwissMapType struct {
	abi.Type
	Key   *abi.Type
	Elem  *abi.Type
	Group *abi.Type // internal type representing a slot group
	// function for hashing keys (ptr to key, seed) -> hash
	Hasher     func(unsafe.Pointer, uintptr) uintptr
	SlotSize   uintptr // size of key/elem slot
	ElemOff    uintptr // offset of elem in key/elem slot
	Flags      uint32
}

// Flag values
const (
	SwissMapNeedKeyUpdate = 1 << iota
	SwissMapHashMightPanic
)

func (mt *SwissMapType) NeedKeyUpdate() bool { // true if we need to update key on an overwrite
	return mt.Flags&SwissMapNeedKeyUpdate != 0
}
func (mt *SwissMapType) HashMightPanic() bool { // true if hash function might panic
	return mt.Flags&SwissMapHashMightPanic != 0
}
