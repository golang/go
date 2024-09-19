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

// Functions below pushed from internal/runtime/maps.

//go:linkname mapaccess1_faststr
func mapaccess1_faststr(t *abi.SwissMapType, m *maps.Map, ky string) unsafe.Pointer

// mapaccess2_faststr should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/ugorji/go/codec
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname mapaccess2_faststr
func mapaccess2_faststr(t *abi.SwissMapType, m *maps.Map, ky string) (unsafe.Pointer, bool)

// mapassign_faststr should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/bytedance/sonic
//   - github.com/ugorji/go/codec
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname mapassign_faststr
func mapassign_faststr(t *abi.SwissMapType, m *maps.Map, s string) unsafe.Pointer

//go:linkname mapdelete_faststr
func mapdelete_faststr(t *abi.SwissMapType, m *maps.Map, ky string)
