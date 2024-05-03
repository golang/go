// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.swissmap

package runtime

import (
	"internal/abi"
	"internal/runtime/maps"
	"unsafe"
)

func mapaccess1_faststr(t *abi.SwissMapType, m *maps.Map, ky string) unsafe.Pointer {
	throw("mapaccess1_faststr unimplemented")
	panic("unreachable")
}

func mapaccess2_faststr(t *abi.SwissMapType, m *maps.Map, ky string) (unsafe.Pointer, bool) {
	throw("mapaccess2_faststr unimplemented")
	panic("unreachable")
}

func mapassign_faststr(t *abi.SwissMapType, m *maps.Map, s string) unsafe.Pointer {
	throw("mapassign_faststr unimplemented")
	panic("unreachable")
}

func mapdelete_faststr(t *abi.SwissMapType, m *maps.Map, ky string) {
	throw("mapdelete_faststr unimplemented")
}
