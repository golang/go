// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Exclude -race because Go's race_darwin_*.syso exports a 2-arg
// __tsan_acquire(ThreadState*, void*) that would collide with C TSAN's
// 1-arg __tsan_acquire(void*).
//
//go:build goexperiment.runtimecgopurego && darwin && !race && !msan

package cgo

import (
	"internal/abi"
	"internal/runtime/atomic"
	"unsafe"
)

func cgoTSanYield()

//go:cgo_import_dynamic _cgo_libc_dyld_image_count _dyld_image_count "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic _cgo_libc_dyld_get_image_name _dyld_get_image_name "/usr/lib/libSystem.B.dylib"

//go:linkname _cgo_libc_dyld_image_count _cgo_libc_dyld_image_count
var _cgo_libc_dyld_image_count byte
var libc_dyld_image_count = unsafe.Pointer(&_cgo_libc_dyld_image_count)

//go:linkname _cgo_libc_dyld_get_image_name _cgo_libc_dyld_get_image_name
var _cgo_libc_dyld_get_image_name byte
var libc_dyld_get_image_name = unsafe.Pointer(&_cgo_libc_dyld_get_image_name)

//go:linknamestd _cgoInitTSan runtime.cgoInitTSan
var _cgoInitTSan = cgoInitTSan

// isTSanRuntimeImage reports whether the C string basename [base, end)
// names a C/C++ TSAN runtime dylib (such as libclang_rt.tsan_osx_dynamic.dylib
// or libtsan.2.dylib).
//
//go:nosplit
//go:norace
//go:nocheckptr
func isTSanRuntimeImage(base, end uintptr) bool {
	n := end - base
	const (
		clangPrefix = "libclang_rt.tsan"
		gccPrefix   = "libtsan."
		dylibSuffix = ".dylib"
	)
	if n < uintptr(len(dylibSuffix)) {
		return false
	}
	for i := uintptr(0); i < uintptr(len(dylibSuffix)); i++ {
		if *(*byte)(unsafe.Pointer(end - uintptr(len(dylibSuffix)) + i)) != dylibSuffix[i] {
			return false
		}
	}
	if n >= uintptr(len(clangPrefix)) {
		match := true
		for i := uintptr(0); i < uintptr(len(clangPrefix)); i++ {
			if *(*byte)(unsafe.Pointer(base + i)) != clangPrefix[i] {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	if n >= uintptr(len(gccPrefix)) {
		for i := uintptr(0); i < uintptr(len(gccPrefix)); i++ {
			if *(*byte)(unsafe.Pointer(base + i)) != gccPrefix[i] {
				return false
			}
		}
		return true
	}
	return false
}

// cgoInitTSan detects C/C++ TSAN hooks during libInit on Darwin, where Mach-O
// __mod_init_func runs go.o's _rt0_*_darwin_lib before _cgo_sanitizer_init.
//
//go:nosplit
//go:norace
//go:nocheckptr
func cgoInitTSan() {
	if atomic.LoadAcquintptr(tsanAcquireFnPtr) != 0 {
		return
	}

	// Call runtime_asmcgocall_no_g directly rather than through asmcgocallNoG
	// to keep libInit's nosplit chain within limits under -N -l.
	var f cgoCallFrame

	// On Darwin, C/C++ TSAN is always loaded as a dynamic runtime library
	// (such as libclang_rt.tsan_*_dynamic.dylib). Scan dyld's already-mapped
	// image list first (skipping image 0, the main executable) so non-TSAN
	// builds avoid dlsym(RTLD_DEFAULT, ...) and a -race host binary cannot
	// match its own 2-arg __tsan_acquire.
	f.fn = uintptr(libc_dyld_image_count)
	runtime_asmcgocall_no_g(cgoCallTrampolinePC(), unsafe.Pointer(&f))
	n := uint32(f.r1)
	foundTSanImage := false
	for i := uint32(1); i < n; i++ {
		f.fn, f.a0, f.r1 = uintptr(libc_dyld_get_image_name), uintptr(i), 0
		runtime_asmcgocall_no_g(cgoCallTrampolinePC(), unsafe.Pointer(&f))
		p := f.r1
		if p == 0 {
			continue
		}
		base := p
		end := p
		for *(*byte)(unsafe.Pointer(end)) != 0 {
			if *(*byte)(unsafe.Pointer(end)) == '/' {
				base = end + 1
			}
			end++
		}
		if isTSanRuntimeImage(base, end) {
			foundTSanImage = true
			break
		}
	}
	if !foundTSanImage {
		return
	}

	const (
		rtldDefault = ^uintptr(1) // RTLD_DEFAULT = (void*)-2 on Darwin
		acqName     = "__tsan_acquire\x00"
		relName     = "__tsan_release\x00"
	)

	f.fn = uintptr(libc_dlsym)
	f.a0 = rtldDefault
	f.a1 = uintptr(unsafe.Pointer(unsafe.StringData(acqName)))
	f.r1 = 0
	runtime_asmcgocall_no_g(cgoCallTrampolinePC(), unsafe.Pointer(&f))
	acq := f.r1
	if acq == 0 {
		// Clear dlerror so a caller checking dlerror() after dlopen()
		// does not observe a spurious missing-symbol error.
		f.fn, f.a0, f.a1 = uintptr(libc_dlerror), 0, 0
		runtime_asmcgocall_no_g(cgoCallTrampolinePC(), unsafe.Pointer(&f))
		return
	}
	f.fn = uintptr(libc_dlsym)
	f.a0 = rtldDefault
	f.a1 = uintptr(unsafe.Pointer(unsafe.StringData(relName)))
	f.r1 = 0
	runtime_asmcgocall_no_g(cgoCallTrampolinePC(), unsafe.Pointer(&f))
	rel := f.r1
	if rel == 0 {
		f.fn, f.a0, f.a1 = uintptr(libc_dlerror), 0, 0
		runtime_asmcgocall_no_g(cgoCallTrampolinePC(), unsafe.Pointer(&f))
		return
	}
	// Populate _cgo_yield before Go runtime startup (so _cgo_sanitizer_init's
	// CAS leaves it untouched) and store release before acquire so a concurrent
	// caller never acquires without releasing.
	atomic.StoreReluintptr(cgoYieldPtr, abi.FuncPCABI0(cgoTSanYield))
	atomic.StoreReluintptr(tsanReleaseFnPtr, rel)
	atomic.StoreReluintptr(tsanAcquireFnPtr, acq)
}
