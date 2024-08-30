// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.swissmap

package runtime

import (
	"unsafe"
)

func mapaccess1_fast32(t *maptype, h *hmap, key uint32) unsafe.Pointer {
	throw("mapaccess1_fast32 unimplemented")
	panic("unreachable")
}

func mapaccess2_fast32(t *maptype, h *hmap, key uint32) (unsafe.Pointer, bool) {
	throw("mapaccess2_fast32 unimplemented")
	panic("unreachable")
}

func mapassign_fast32(t *maptype, h *hmap, key uint32) unsafe.Pointer {
	throw("mapassign_fast32 unimplemented")
	panic("unreachable")
}

func mapassign_fast32ptr(t *maptype, h *hmap, key unsafe.Pointer) unsafe.Pointer {
	throw("mapassign_fast32ptr unimplemented")
	panic("unreachable")
}

func mapdelete_fast32(t *maptype, h *hmap, key uint32) {
	throw("mapdelete_fast32 unimplemented")
}
