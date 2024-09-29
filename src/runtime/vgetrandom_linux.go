// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (amd64 || arm64 || arm64be || ppc64 || ppc64le || loong64 || s390x)

package runtime

import (
	"internal/cpu"
	"unsafe"
)

func vgetrandom1(buf *byte, length uintptr, flags uint32, state uintptr, stateSize uintptr) int

var vgetrandomAlloc struct {
	states     []uintptr
	statesLock mutex
	stateSize  uintptr
	mmapProt   int32
	mmapFlags  int32
}

func vgetrandomInit() {
	if vdsoGetrandomSym == 0 {
		return
	}

	var params struct {
		SizeOfOpaqueState uint32
		MmapProt          uint32
		MmapFlags         uint32
		reserved          [13]uint32
	}
	if vgetrandom1(nil, 0, 0, uintptr(unsafe.Pointer(&params)), ^uintptr(0)) != 0 {
		return
	}
	vgetrandomAlloc.stateSize = uintptr(params.SizeOfOpaqueState)
	vgetrandomAlloc.mmapProt = int32(params.MmapProt)
	vgetrandomAlloc.mmapFlags = int32(params.MmapFlags)

	lockInit(&vgetrandomAlloc.statesLock, lockRankLeafRank)
}

func vgetrandomGetState() uintptr {
	lock(&vgetrandomAlloc.statesLock)
	if len(vgetrandomAlloc.states) == 0 {
		num := uintptr(ncpu) // Just a reasonable size hint to start.
		stateSizeCacheAligned := (vgetrandomAlloc.stateSize + cpu.CacheLineSize - 1) &^ (cpu.CacheLineSize - 1)
		allocSize := (num*stateSizeCacheAligned + physPageSize - 1) &^ (physPageSize - 1)
		num = (physPageSize / stateSizeCacheAligned) * (allocSize / physPageSize)
		p, err := mmap(nil, allocSize, vgetrandomAlloc.mmapProt, vgetrandomAlloc.mmapFlags, -1, 0)
		if err != 0 {
			unlock(&vgetrandomAlloc.statesLock)
			return 0
		}
		newBlock := uintptr(p)
		if vgetrandomAlloc.states == nil {
			vgetrandomAlloc.states = make([]uintptr, 0, num)
		}
		for i := uintptr(0); i < num; i++ {
			if (newBlock&(physPageSize-1))+vgetrandomAlloc.stateSize > physPageSize {
				newBlock = (newBlock + physPageSize - 1) &^ (physPageSize - 1)
			}
			vgetrandomAlloc.states = append(vgetrandomAlloc.states, newBlock)
			newBlock += stateSizeCacheAligned
		}
	}
	state := vgetrandomAlloc.states[len(vgetrandomAlloc.states)-1]
	vgetrandomAlloc.states = vgetrandomAlloc.states[:len(vgetrandomAlloc.states)-1]
	unlock(&vgetrandomAlloc.statesLock)
	return state
}

func vgetrandomPutState(state uintptr) {
	lock(&vgetrandomAlloc.statesLock)
	vgetrandomAlloc.states = append(vgetrandomAlloc.states, state)
	unlock(&vgetrandomAlloc.statesLock)
}

// This is exported for use in internal/syscall/unix as well as x/sys/unix.
//
//go:linkname vgetrandom
func vgetrandom(p []byte, flags uint32) (ret int, supported bool) {
	if vgetrandomAlloc.stateSize == 0 {
		return -1, false
	}

	mp := acquirem()
	if mp.vgetrandomState == 0 {
		state := vgetrandomGetState()
		if state == 0 {
			releasem(mp)
			return -1, false
		}
		mp.vgetrandomState = state
	}

	ret = vgetrandom1(unsafe.SliceData(p), uintptr(len(p)), flags, mp.vgetrandomState, vgetrandomAlloc.stateSize)
	supported = true

	releasem(mp)
	return
}
