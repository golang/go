// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import (
	"internal/abi"
	"runtime"
	"strconv"
	"unsafe"
)

func AndroidVersion() int {
	const PROP_VALUE_MAX = 92
	var value [PROP_VALUE_MAX]byte
	name := []byte("ro.build.version.release\x00")
	length := __system_property_get(&name[0], &value[0])
	for i := int32(0); i < length; i++ {
		if value[i] < '0' || value[i] > '9' {
			length = i
			break
		}
	}
	version, _ := strconv.Atoi(unsafe.String(&value[0], length))
	return version
}

// The *_trampoline functions convert from the Go calling convention to the C calling convention
// and then call the underlying libc function. These are defined in sys_android_$ARCH.s.

//go:nosplit
//go:cgo_unsafe_args
func __system_property_get(name *byte, value *byte) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(__system_property_get_trampoline)), unsafe.Pointer(&name))
	runtime.KeepAlive(name)
	runtime.KeepAlive(value)
	return ret
}
func __system_property_get_trampoline()

//go:linkname libcCall runtime.libcCall
//go:noescape
func libcCall(fn, arg unsafe.Pointer) int32

// Tell the linker that the libc_* functions are to be found
// in a system library, with the libc_ prefix missing.

//go:cgo_import_dynamic libc___system_property_get __system_property_get "libc.so"
