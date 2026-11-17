// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"internal/goexperiment"
	"unsafe"
)

// logosCgoPointerFreeAddr reports whether the existing cgo address marker
// restricts the check to a pointer-free value. Both type arguments are non-nil.
// Such a value makes cgoCheckArg return without inspecting the address.
func logosCgoPointerFreeAddr(t, marker *_type) bool {
	return goexperiment.LogosCgoOpt &&
		t.Kind() == abi.Pointer &&
		marker.Kind() == abi.Bool &&
		!(*ptrtype)(unsafe.Pointer(t)).Elem.Pointers()
}
