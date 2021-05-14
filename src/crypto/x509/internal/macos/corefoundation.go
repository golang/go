// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin && !ios
// +build darwin,!ios

// Package macOS provides cgo-less wrappers for Core Foundation and
// Security.framework, similarly to how package syscall provides access to
// libSystem.dylib.
package macOS

import (
	"errors"
	"internal/abi"
	"reflect"
	"runtime"
	"unsafe"
)

// Core Foundation linker flags for the external linker. See Issue 42459.
//go:cgo_ldflag "-framework"
//go:cgo_ldflag "CoreFoundation"

// CFRef is an opaque reference to a Core Foundation object. It is a pointer,
// but to memory not owned by Go, so not an unsafe.Pointer.
type CFRef uintptr

// CFDataToSlice returns a copy of the contents of data as a bytes slice.
func CFDataToSlice(data CFRef) []byte {
	length := CFDataGetLength(data)
	ptr := CFDataGetBytePtr(data)
	src := (*[1 << 20]byte)(unsafe.Pointer(ptr))[:length:length]
	out := make([]byte, length)
	copy(out, src)
	return out
}

type CFString CFRef

const kCFAllocatorDefault = 0
const kCFStringEncodingUTF8 = 0x08000100

//go:cgo_import_dynamic x509_CFStringCreateWithBytes CFStringCreateWithBytes "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation"

// StringToCFString returns a copy of the UTF-8 contents of s as a new CFString.
func StringToCFString(s string) CFString {
	p := unsafe.Pointer((*reflect.StringHeader)(unsafe.Pointer(&s)).Data)
	ret := syscall(abi.FuncPCABI0(x509_CFStringCreateWithBytes_trampoline), kCFAllocatorDefault, uintptr(p),
		uintptr(len(s)), uintptr(kCFStringEncodingUTF8), 0 /* isExternalRepresentation */, 0)
	runtime.KeepAlive(p)
	return CFString(ret)
}
func x509_CFStringCreateWithBytes_trampoline()

//go:cgo_import_dynamic x509_CFDictionaryGetValueIfPresent CFDictionaryGetValueIfPresent "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation"

func CFDictionaryGetValueIfPresent(dict CFRef, key CFString) (value CFRef, ok bool) {
	ret := syscall(abi.FuncPCABI0(x509_CFDictionaryGetValueIfPresent_trampoline), uintptr(dict), uintptr(key),
		uintptr(unsafe.Pointer(&value)), 0, 0, 0)
	if ret == 0 {
		return 0, false
	}
	return value, true
}
func x509_CFDictionaryGetValueIfPresent_trampoline()

const kCFNumberSInt32Type = 3

//go:cgo_import_dynamic x509_CFNumberGetValue CFNumberGetValue "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation"

func CFNumberGetValue(num CFRef) (int32, error) {
	var value int32
	ret := syscall(abi.FuncPCABI0(x509_CFNumberGetValue_trampoline), uintptr(num), uintptr(kCFNumberSInt32Type),
		uintptr(unsafe.Pointer(&value)), 0, 0, 0)
	if ret == 0 {
		return 0, errors.New("CFNumberGetValue call failed")
	}
	return value, nil
}
func x509_CFNumberGetValue_trampoline()

//go:cgo_import_dynamic x509_CFDataGetLength CFDataGetLength "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation"

func CFDataGetLength(data CFRef) int {
	ret := syscall(abi.FuncPCABI0(x509_CFDataGetLength_trampoline), uintptr(data), 0, 0, 0, 0, 0)
	return int(ret)
}
func x509_CFDataGetLength_trampoline()

//go:cgo_import_dynamic x509_CFDataGetBytePtr CFDataGetBytePtr "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation"

func CFDataGetBytePtr(data CFRef) uintptr {
	ret := syscall(abi.FuncPCABI0(x509_CFDataGetBytePtr_trampoline), uintptr(data), 0, 0, 0, 0, 0)
	return ret
}
func x509_CFDataGetBytePtr_trampoline()

//go:cgo_import_dynamic x509_CFArrayGetCount CFArrayGetCount "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation"

func CFArrayGetCount(array CFRef) int {
	ret := syscall(abi.FuncPCABI0(x509_CFArrayGetCount_trampoline), uintptr(array), 0, 0, 0, 0, 0)
	return int(ret)
}
func x509_CFArrayGetCount_trampoline()

//go:cgo_import_dynamic x509_CFArrayGetValueAtIndex CFArrayGetValueAtIndex "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation"

func CFArrayGetValueAtIndex(array CFRef, index int) CFRef {
	ret := syscall(abi.FuncPCABI0(x509_CFArrayGetValueAtIndex_trampoline), uintptr(array), uintptr(index), 0, 0, 0, 0)
	return CFRef(ret)
}
func x509_CFArrayGetValueAtIndex_trampoline()

//go:cgo_import_dynamic x509_CFEqual CFEqual "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation"

func CFEqual(a, b CFRef) bool {
	ret := syscall(abi.FuncPCABI0(x509_CFEqual_trampoline), uintptr(a), uintptr(b), 0, 0, 0, 0)
	return ret == 1
}
func x509_CFEqual_trampoline()

//go:cgo_import_dynamic x509_CFRelease CFRelease "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation"

func CFRelease(ref CFRef) {
	syscall(abi.FuncPCABI0(x509_CFRelease_trampoline), uintptr(ref), 0, 0, 0, 0, 0)
}
func x509_CFRelease_trampoline()

// syscall is implemented in the runtime package (runtime/sys_darwin.go)
func syscall(fn, a1, a2, a3, a4, a5, a6 uintptr) uintptr
