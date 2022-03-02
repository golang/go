// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin

// Package macOS provides cgo-less wrappers for Core Foundation and
// Security.framework, similarly to how package syscall provides access to
// libSystem.dylib.
package macOS

import (
	"errors"
	"internal/abi"
	"reflect"
	"runtime"
	"time"
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

// CFStringToString returns a Go string representation of the passed
// in CFString.
func CFStringToString(ref CFRef) string {
	data := CFStringCreateExternalRepresentation(ref)
	b := CFDataToSlice(data)
	CFRelease(data)
	return string(b)
}

// TimeToCFDateRef converts a time.Time into an apple CFDateRef
func TimeToCFDateRef(t time.Time) CFRef {
	secs := t.Sub(time.Date(2001, 1, 1, 0, 0, 0, 0, time.UTC)).Seconds()
	ref := CFDateCreate(secs)
	return ref
}

type CFString CFRef

const kCFAllocatorDefault = 0
const kCFStringEncodingUTF8 = 0x08000100

//go:cgo_import_dynamic x509_CFDataCreate CFDataCreate "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation"

func BytesToCFData(b []byte) CFRef {
	p := unsafe.Pointer((*reflect.SliceHeader)(unsafe.Pointer(&b)).Data)
	ret := syscall(abi.FuncPCABI0(x509_CFDataCreate_trampoline), kCFAllocatorDefault, uintptr(p), uintptr(len(b)), 0, 0, 0)
	runtime.KeepAlive(p)
	return CFRef(ret)
}
func x509_CFDataCreate_trampoline()

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

//go:cgo_import_dynamic x509_CFArrayCreateMutable CFArrayCreateMutable "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation"

func CFArrayCreateMutable() CFRef {
	ret := syscall(abi.FuncPCABI0(x509_CFArrayCreateMutable_trampoline), kCFAllocatorDefault, 0, 0 /* kCFTypeArrayCallBacks */, 0, 0, 0)
	return CFRef(ret)
}
func x509_CFArrayCreateMutable_trampoline()

//go:cgo_import_dynamic x509_CFArrayAppendValue CFArrayAppendValue "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation"

func CFArrayAppendValue(array CFRef, val CFRef) {
	syscall(abi.FuncPCABI0(x509_CFArrayAppendValue_trampoline), uintptr(array), uintptr(val), 0, 0, 0, 0)
}
func x509_CFArrayAppendValue_trampoline()

//go:cgo_import_dynamic x509_CFDateCreate CFDateCreate "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation"

func CFDateCreate(seconds float64) CFRef {
	ret := syscall(abi.FuncPCABI0(x509_CFDateCreate_trampoline), kCFAllocatorDefault, 0, 0, 0, 0, seconds)
	return CFRef(ret)
}
func x509_CFDateCreate_trampoline()

//go:cgo_import_dynamic x509_CFErrorCopyDescription CFErrorCopyDescription "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation"

func CFErrorCopyDescription(errRef CFRef) CFRef {
	ret := syscall(abi.FuncPCABI0(x509_CFErrorCopyDescription_trampoline), uintptr(errRef), 0, 0, 0, 0, 0)
	return CFRef(ret)
}
func x509_CFErrorCopyDescription_trampoline()

//go:cgo_import_dynamic x509_CFStringCreateExternalRepresentation CFStringCreateExternalRepresentation "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation"

func CFStringCreateExternalRepresentation(strRef CFRef) CFRef {
	ret := syscall(abi.FuncPCABI0(x509_CFStringCreateExternalRepresentation_trampoline), kCFAllocatorDefault, uintptr(strRef), kCFStringEncodingUTF8, 0, 0, 0)
	return CFRef(ret)
}
func x509_CFStringCreateExternalRepresentation_trampoline()

// syscall is implemented in the runtime package (runtime/sys_darwin.go)
func syscall(fn, a1, a2, a3, a4, a5 uintptr, f1 float64) uintptr

// ReleaseCFArray iterates through an array, releasing its contents, and then
// releases the array itself. This is necessary because we cannot, easily, set the
// CFArrayCallBacks argument when creating CFArrays.
func ReleaseCFArray(array CFRef) {
	for i := 0; i < CFArrayGetCount(array); i++ {
		ref := CFArrayGetValueAtIndex(array, i)
		CFRelease(ref)
	}
	CFRelease(array)
}
