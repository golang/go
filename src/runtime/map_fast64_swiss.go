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

func mapaccess1_fast64(t *abi.SwissMapType, m *maps.Map, key uint64) unsafe.Pointer {
	throw("mapaccess1_fast64 unimplemented")
	panic("unreachable")
}

func mapaccess2_fast64(t *abi.SwissMapType, m *maps.Map, key uint64) (unsafe.Pointer, bool) {
	throw("mapaccess2_fast64 unimplemented")
	panic("unreachable")
}

func mapassign_fast64(t *abi.SwissMapType, m *maps.Map, key uint64) unsafe.Pointer {
	throw("mapassign_fast64 unimplemented")
	panic("unreachable")
}

func mapassign_fast64ptr(t *abi.SwissMapType, m *maps.Map, key unsafe.Pointer) unsafe.Pointer {
	throw("mapassign_fast64ptr unimplemented")
	panic("unreachable")
}

func mapdelete_fast64(t *abi.SwissMapType, m *maps.Map, key uint64) {
	throw("mapdelete_fast64 unimplemented")
}
