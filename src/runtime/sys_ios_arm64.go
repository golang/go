// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"unsafe"
)

// CoreFoundation linker flags for the external linker.
//
//go:cgo_ldflag "-framework"
//go:cgo_ldflag "CoreFoundation"

//go:nosplit
func chdir(path *byte) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(chdir_trampoline)), unsafe.Pointer(&path))
	KeepAlive(path)
	return ret
}
func chdir_trampoline()

//go:nosplit
func cfBundleGetMainBundle() (bundle uintptr) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(cfBundleGetMainBundle_trampoline)), unsafe.Pointer(&bundle))
	return bundle
}
func cfBundleGetMainBundle_trampoline()

//go:nosplit
func cfBundleCopyBundleURL(bundle uintptr) uintptr {
	args := struct {
		bundle uintptr
		ret    uintptr
	}{bundle: bundle}
	libcCall(unsafe.Pointer(abi.FuncPCABI0(cfBundleCopyBundleURL_trampoline)), unsafe.Pointer(&args))
	return args.ret
}
func cfBundleCopyBundleURL_trampoline()

//go:nosplit
func cfURLGetFileSystemRepresentation(url uintptr, resolveAgainstBase bool, path *byte, pathLen uintptr) bool {
	args := struct {
		url, resolveAgainstBase uintptr
		path                    *byte
		pathLen                 uintptr
		ret                     uintptr
	}{
		url:     url,
		path:    path,
		pathLen: pathLen,
	}
	if resolveAgainstBase {
		args.resolveAgainstBase = 1
	}
	libcCall(unsafe.Pointer(abi.FuncPCABI0(cfURLGetFileSystemRepresentation_trampoline)), unsafe.Pointer(&args))
	KeepAlive(path)
	return args.ret != 0
}
func cfURLGetFileSystemRepresentation_trampoline()

//go:nosplit
func cfStringCreateWithCString(alloc uintptr, str *byte, encoding uintptr) uintptr {
	args := struct {
		alloc    uintptr
		str      *byte
		encoding uintptr
		ret      uintptr
	}{
		alloc:    alloc,
		str:      str,
		encoding: encoding,
	}
	libcCall(unsafe.Pointer(abi.FuncPCABI0(cfStringCreateWithCString_trampoline)), unsafe.Pointer(&args))
	KeepAlive(str)
	return args.ret
}
func cfStringCreateWithCString_trampoline()

//go:nosplit
func cfBundleGetValueForInfoDictionaryKey(bundle, key uintptr) uintptr {
	args := struct {
		bundle uintptr
		key    uintptr
		ret    uintptr
	}{
		bundle: bundle,
		key:    key,
	}
	libcCall(unsafe.Pointer(abi.FuncPCABI0(cfBundleGetValueForInfoDictionaryKey_trampoline)), unsafe.Pointer(&args))
	return args.ret
}
func cfBundleGetValueForInfoDictionaryKey_trampoline()

//go:nosplit
func cfStringGetCString(str uintptr, buf *byte, bufLen uintptr, encoding uintptr) bool {
	args := struct {
		str      uintptr
		buf      *byte
		bufLen   uintptr
		encoding uintptr
		ret      uintptr
	}{
		str:      str,
		buf:      buf,
		bufLen:   bufLen,
		encoding: encoding,
	}
	libcCall(unsafe.Pointer(abi.FuncPCABI0(cfStringGetCString_trampoline)), unsafe.Pointer(&args))
	KeepAlive(buf)
	return args.ret != 0
}
func cfStringGetCString_trampoline()

//go:nosplit
func cfRelease(ref uintptr) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(cfRelease_trampoline)), unsafe.Pointer(&ref))
}
func cfRelease_trampoline()

//go:cgo_import_dynamic libc_chdir chdir "/usr/lib/libSystem.B.dylib"

//go:cgo_import_dynamic libc_CFBundleGetMainBundle CFBundleGetMainBundle "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation"
//go:cgo_import_dynamic libc_CFBundleCopyBundleURL CFBundleCopyBundleURL "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation"
//go:cgo_import_dynamic libc_CFURLGetFileSystemRepresentation CFURLGetFileSystemRepresentation "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation"
//go:cgo_import_dynamic libc_CFStringCreateWithCString CFStringCreateWithCString "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation"
//go:cgo_import_dynamic libc_CFBundleGetValueForInfoDictionaryKey CFBundleGetValueForInfoDictionaryKey "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation"
//go:cgo_import_dynamic libc_CFStringGetCString CFStringGetCString "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation"
//go:cgo_import_dynamic libc_CFRelease CFRelease "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation"
