// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

var wasmContCur, wasmContNext, wasmContPaused int64

var wasmConts struct {
	tab  uintptr
	size uintptr
	n    uintptr
}

type wasmContEntry struct {
	g    uintptr
	slot int64
}

//go:nosplit
func wasmContSlot(gp *g) int64 {
	if wasmConts.n*2 >= wasmConts.size {
		wasmContGrow()
	}
	key := uintptr(unsafe.Pointer(gp))
	e := wasmContFind(wasmConts.tab, wasmConts.size, key)
	if e.g == 0 {
		e.g = key
		wasmConts.n++
		e.slot = int64(wasmConts.n)
	}
	return e.slot
}

//go:nosplit
func wasmContFind(tab, size, key uintptr) *wasmContEntry {
	i := (key >> 3) * 0x9e3779b97f4a7c15 & (size - 1)
	for {
		e := (*wasmContEntry)(unsafe.Pointer(tab + i*unsafe.Sizeof(wasmContEntry{})))
		if e.g == key || e.g == 0 {
			return e
		}
		i = (i + 1) & (size - 1)
	}
}

func wasmContGrow() {
	size := max(wasmConts.size*2, 64)
	tab := uintptr(persistentalloc(size*unsafe.Sizeof(wasmContEntry{}), 8, &memstats.other_sys))
	for i := uintptr(0); i < wasmConts.size; i++ {
		old := (*wasmContEntry)(unsafe.Pointer(wasmConts.tab + i*unsafe.Sizeof(wasmContEntry{})))
		if old.g != 0 {
			*wasmContFind(tab, size, old.g) = *old
		}
	}
	wasmConts.tab, wasmConts.size = tab, size
}
