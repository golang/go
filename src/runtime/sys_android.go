// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"unsafe"
)

// The *_trampoline functions convert from the Go calling convention to the C calling convention
// and then call the underlying libc function. These are defined in sys_android_$ARCH.s.

//go:nosplit
//go:cgo_unsafe_args
func __system_property_get(name *byte, value *byte) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(__system_property_get_trampoline)), unsafe.Pointer(&name))
	KeepAlive(name)
	KeepAlive(value)
	return ret
}
func __system_property_get_trampoline()

// Tell the linker that the libc_* functions are to be found
// in a system library, with the libc_ prefix missing.

//go:cgo_import_dynamic libc___system_property_get __system_property_get "libc.so"
