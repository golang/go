// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.swissmap

// This file allows non-GOEXPERIMENT=swissmap builds (i.e., old map builds) to
// construct a swissmap table for running the tests in this package.

package maps

import (
	"internal/abi"
	"unsafe"
)

type instantiatedGroup[K comparable, V any] struct {
	ctrls ctrlGroup
	slots [abi.SwissMapGroupSlots]instantiatedSlot[K, V]
}

type instantiatedSlot[K comparable, V any] struct {
	key  K
	elem V
}

func newTestMapType[K comparable, V any]() *abi.SwissMapType {
	var m map[K]V
	mTyp := abi.TypeOf(m)
	omt := (*abi.OldMapType)(unsafe.Pointer(mTyp))

	var grp instantiatedGroup[K, V]
	var slot instantiatedSlot[K, V]

	mt := &abi.SwissMapType{
		Key:       omt.Key,
		Elem:      omt.Elem,
		Group:     abi.TypeOf(grp),
		Hasher:    omt.Hasher,
		SlotSize:  unsafe.Sizeof(slot),
		GroupSize: unsafe.Sizeof(grp),
		ElemOff:   unsafe.Offsetof(slot.elem),
	}
	if omt.NeedKeyUpdate() {
		mt.Flags |= abi.SwissMapNeedKeyUpdate
	}
	if omt.HashMightPanic() {
		mt.Flags |= abi.SwissMapHashMightPanic
	}
	return mt
}
