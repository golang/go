// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && android

package cgo

import "unsafe"

//go:linkname _cgo_libc_dlclose _cgo_libc_dlclose
var _cgo_libc_dlclose byte
var libc_dlclose = unsafe.Pointer(&_cgo_libc_dlclose)

//go:linkname _cgo_libc_android_log_write _cgo_libc_android_log_write
var _cgo_libc_android_log_write byte
var libc_android_log_write = unsafe.Pointer(&_cgo_libc_android_log_write)

// cgoInitTLS allocates or verifies a thread-local storage slot for g on Android.
//
//go:nosplit
//go:norace
//go:nocheckptr
func cgoInitTLS(tlsgAddr, tlsbase uintptr) {
	// Truncated to a different magic value on 32-bit; that's ok.
	const magic1 = uintptr(0x23581321345589 & (1<<(8*unsafe.Sizeof(uintptr(0))) - 1))
	tlsg := (*uintptr)(unsafe.Pointer(tlsgAddr))

	var done bool
	var msg string
	if done, msg = cgoInitTLSQ(tlsg); done {
		if msg != "" {
			cgoAndroidFatal(msg)
		}
		return
	}

	if msg = cgoInitTLSKey(tlsg, tlsbase, magic1); msg != "" {
		cgoAndroidFatal(msg)
	}
}

//go:nosplit
//go:norace
//go:nocheckptr
func cgoInitTLSQ(tlsg *uintptr) (bool, string) {
	const _TLS_SLOT_APP = 2
	const _RTLD_LAZY = 1 // from bionic's <dlfcn.h>
	// Check for Android Q where we can use the free TLS_SLOT_APP slot.
	const libcName = "libc.so\x00"
	handle := asmcgocallNoG(uintptr(libc_dlopen), uintptr(unsafe.Pointer(unsafe.StringData(libcName))), _RTLD_LAZY, 0, 0, 0, 0)
	if handle == 0 {
		return true, "runtime/cgo: inittls: failed to dlopen main program\n\x00"
	}
	// android_get_device_api_level is introduced in Android Q, so its mere presence
	// is enough.
	const symName = "android_get_device_api_level\x00"
	getVer := asmcgocallNoG(uintptr(libc_dlsym), handle, uintptr(unsafe.Pointer(unsafe.StringData(symName))), 0, 0, 0, 0)
	asmcgocallNoG(uintptr(libc_dlclose), handle, 0, 0, 0, 0, 0)
	if getVer == 0 {
		return false, ""
	}
	const off = _TLS_SLOT_APP * unsafe.Sizeof(uintptr(0))
	// tlsg is initialized to Q's free TLS slot. Verify it while we're here.
	if *tlsg != off {
		return true, "runtime/cgo: tlsg offset wrong\n\x00"
	}
	return true, ""
}

//go:nosplit
//go:norace
//go:nocheckptr
func cgoInitTLSKey(tlsg *uintptr, tlsbase, magic1 uintptr) string {
	var k uint32 // Bionic pthread_key_t is int
	err := int32(asmcgocallNoG(uintptr(libc_pthread_key_create), uintptr(unsafe.Pointer(&k)), 0, 0, 0, 0, 0))
	if err != 0 {
		return "runtime/cgo: pthread_key_create failed\n\x00"
	}
	asmcgocallNoG(uintptr(libc_pthread_setspecific), uintptr(k), magic1, 0, 0, 0, 0)
	// If thread local slots are laid out as we expect, our magic word will
	// be located at some low offset from tlsbase. However, just in case something went
	// wrong, the search is limited to sensible offsets. PTHREAD_KEYS_MAX was the
	// original limit, but issue 19472 made a higher limit necessary.
	const ptrSize = unsafe.Sizeof(uintptr(0))
	for i := uintptr(0); i < 384; i++ {
		if *(*uintptr)(unsafe.Pointer(tlsbase + i*ptrSize)) == magic1 {
			*tlsg = i * ptrSize
			asmcgocallNoG(uintptr(libc_pthread_setspecific), uintptr(k), 0, 0, 0, 0, 0)
			return ""
		}
	}
	return "runtime/cgo: inittls: could not find pthread key\n\x00"
}

//go:nosplit
//go:norace
//go:nocheckptr
func cgoAndroidFatal(msg string) {
	// msg must end with "\n\x00".
	// When running from an Android .apk, /dev/stderr and /dev/stdout
	// redirect to /dev/null. And when running a test binary
	// via adb shell, it's easy to miss logcat. So write to both.
	// Write to logcat first: runtime.writeErrStr calls getg after
	// writing to stderr, which is not safe before the TLS slot for g
	// has been set up by cgoInitTLS.
	const _ANDROID_LOG_FATAL = 7
	const tag = "runtime/cgo\x00"
	asmcgocallNoG(uintptr(libc_android_log_write), _ANDROID_LOG_FATAL,
		uintptr(unsafe.Pointer(unsafe.StringData(tag))),
		uintptr(unsafe.Pointer(unsafe.StringData(msg))), 0, 0, 0)
	runtime_writeErrStr(msg[:len(msg)-1])
	runtime_abort()
}

// cgoFatal reports a fatal runtime/cgo error and aborts.
// msg must end with "\n\x00".
//
//go:nosplit
//go:norace
//go:nocheckptr
func cgoFatal(msg string) {
	cgoAndroidFatal(msg)
}

//go:cgo_ldflag "-ldl"
//go:cgo_ldflag "-llog"

//go:cgo_import_dynamic _ _ "libc.so"
//go:cgo_import_dynamic _ _ "libdl.so"
//go:cgo_import_dynamic _ _ "liblog.so"
//go:cgo_import_dynamic _cgo_libc_dlopen dlopen "libdl.so"
//go:cgo_import_dynamic _cgo_libc_dlsym dlsym "libdl.so"
//go:cgo_import_dynamic _cgo_libc_dlclose dlclose "libdl.so"
//go:cgo_import_dynamic _cgo_libc_dlerror dlerror "libdl.so"
//go:cgo_import_dynamic _cgo_libc_android_log_write __android_log_write "liblog.so"
//go:cgo_import_dynamic _cgo_libc_mmap mmap "libc.so"
//go:cgo_import_dynamic _cgo_libc_munmap munmap "libc.so"
//go:cgo_import_dynamic _cgo_libc_sigaction sigaction "libc.so"
//go:cgo_import_dynamic _cgo_libc_sigfillset sigfillset "libc.so"
//go:cgo_import_dynamic _cgo_libc_errno_location __errno "libc.so"
//go:cgo_import_dynamic _cgo_libc_setenv setenv "libc.so"
//go:cgo_import_dynamic _cgo_libc_unsetenv unsetenv "libc.so"
//go:cgo_import_dynamic _cgo_libc_clearenv clearenv "libc.so"
//go:cgo_import_dynamic _cgo_libc_pthread_create pthread_create "libc.so"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_init pthread_attr_init "libc.so"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_setdetachstate pthread_attr_setdetachstate "libc.so"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_getstacksize pthread_attr_getstacksize "libc.so"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_destroy pthread_attr_destroy "libc.so"
//go:cgo_import_dynamic _cgo_libc_pthread_key_create pthread_key_create "libc.so"
//go:cgo_import_dynamic _cgo_libc_pthread_setspecific pthread_setspecific "libc.so"
//go:cgo_import_dynamic _cgo_libc_pthread_self pthread_self "libc.so"
//go:cgo_import_dynamic _cgo_libc_pthread_getattr_np pthread_getattr_np "libc.so"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_getstack pthread_attr_getstack "libc.so"
//go:cgo_import_dynamic _cgo_libc_pthread_sigmask pthread_sigmask "libc.so"
//go:cgo_import_dynamic _cgo_libc_setegid setegid "libc.so"
//go:cgo_import_dynamic _cgo_libc_seteuid seteuid "libc.so"
//go:cgo_import_dynamic _cgo_libc_setgid setgid "libc.so"
//go:cgo_import_dynamic _cgo_libc_setgroups setgroups "libc.so"
//go:cgo_import_dynamic _cgo_libc_setregid setregid "libc.so"
//go:cgo_import_dynamic _cgo_libc_setresgid setresgid "libc.so"
//go:cgo_import_dynamic _cgo_libc_setresuid setresuid "libc.so"
//go:cgo_import_dynamic _cgo_libc_setreuid setreuid "libc.so"
//go:cgo_import_dynamic _cgo_libc_setuid setuid "libc.so"
