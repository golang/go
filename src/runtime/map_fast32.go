// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"internal/runtime/maps"
	"unsafe"
)

// Functions below pushed from internal/runtime/maps.

//go:linkname mapaccess1_fast32
func mapaccess1_fast32(t *abi.MapType, m *maps.Map, key uint32) unsafe.Pointer

// mapaccess2_fast32 should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/ugorji/go/codec
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname mapaccess2_fast32
func mapaccess2_fast32(t *abi.MapType, m *maps.Map, key uint32) (unsafe.Pointer, bool)

// mapassign_fast32 should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/bytedance/sonic
//   - github.com/ugorji/go/codec
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname mapassign_fast32
func mapassign_fast32(t *abi.MapType, m *maps.Map, key uint32) unsafe.Pointer

// mapassign_fast32ptr should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/ugorji/go/codec
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname mapassign_fast32ptr
func mapassign_fast32ptr(t *abi.MapType, m *maps.Map, key unsafe.Pointer) unsafe.Pointer

//go:linkname mapdelete_fast32
func mapdelete_fast32(t *abi.MapType, m *maps.Map, key uint32)
