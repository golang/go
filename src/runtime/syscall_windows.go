// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"unsafe"
)

type callbacks struct {
	lock mutex
	ctxt [cb_max]*wincallbackcontext
	n    int
}

func (c *wincallbackcontext) isCleanstack() bool {
	return c.cleanstack
}

func (c *wincallbackcontext) setCleanstack(cleanstack bool) {
	c.cleanstack = cleanstack
}

var (
	cbs     callbacks
	cbctxts **wincallbackcontext = &cbs.ctxt[0] // to simplify access to cbs.ctxt in sys_windows_*.s
)

func callbackasm()

// callbackasmAddr returns address of runtime.callbackasm
// function adjusted by i.
// On x86 and amd64, runtime.callbackasm is a series of CALL instructions,
// and we want callback to arrive at
// correspondent call instruction instead of start of
// runtime.callbackasm.
// On ARM, runtime.callbackasm is a series of mov and branch instructions.
// R12 is loaded with the callback index. Each entry is two instructions,
// hence 8 bytes.
func callbackasmAddr(i int) uintptr {
	var entrySize int
	switch GOARCH {
	default:
		panic("unsupported architecture")
	case "386", "amd64":
		entrySize = 5
	case "arm":
		// On ARM, each entry is a MOV instruction
		// followed by a branch instruction
		entrySize = 8
	}
	return funcPC(callbackasm) + uintptr(i*entrySize)
}

//go:linkname compileCallback syscall.compileCallback
func compileCallback(fn eface, cleanstack bool) (code uintptr) {
	if fn._type == nil || (fn._type.kind&kindMask) != kindFunc {
		panic("compileCallback: expected function with one uintptr-sized result")
	}
	ft := (*functype)(unsafe.Pointer(fn._type))
	if len(ft.out()) != 1 {
		panic("compileCallback: expected function with one uintptr-sized result")
	}
	uintptrSize := unsafe.Sizeof(uintptr(0))
	if ft.out()[0].size != uintptrSize {
		panic("compileCallback: expected function with one uintptr-sized result")
	}
	argsize := uintptr(0)
	for _, t := range ft.in() {
		if t.size > uintptrSize {
			panic("compileCallback: argument size is larger than uintptr")
		}
		argsize += uintptrSize
	}

	lock(&cbs.lock)
	defer unlock(&cbs.lock)

	n := cbs.n
	for i := 0; i < n; i++ {
		if cbs.ctxt[i].gobody == fn.data && cbs.ctxt[i].isCleanstack() == cleanstack {
			return callbackasmAddr(i)
		}
	}
	if n >= cb_max {
		throw("too many callback functions")
	}

	c := new(wincallbackcontext)
	c.gobody = fn.data
	c.argsize = argsize
	c.setCleanstack(cleanstack)
	if cleanstack && argsize != 0 {
		c.restorestack = argsize
	} else {
		c.restorestack = 0
	}
	cbs.ctxt[n] = c
	cbs.n++

	return callbackasmAddr(n)
}

const _LOAD_LIBRARY_SEARCH_SYSTEM32 = 0x00000800

//go:linkname syscall_loadsystemlibrary syscall.loadsystemlibrary
//go:nosplit
func syscall_loadsystemlibrary(filename *uint16) (handle, err uintptr) {
	lockOSThread()
	defer unlockOSThread()
	c := &getg().m.syscall

	if useLoadLibraryEx {
		c.fn = getLoadLibraryEx()
		c.n = 3
		args := struct {
			lpFileName *uint16
			hFile      uintptr // always 0
			flags      uint32
		}{filename, 0, _LOAD_LIBRARY_SEARCH_SYSTEM32}
		c.args = uintptr(noescape(unsafe.Pointer(&args)))
	} else {
		// User doesn't have KB2533623 installed. The caller
		// wanted to only load the filename DLL from the
		// System32 directory but that facility doesn't exist,
		// so just load it the normal way. This is a potential
		// security risk, but so is not installing security
		// updates.
		c.fn = getLoadLibrary()
		c.n = 1
		c.args = uintptr(noescape(unsafe.Pointer(&filename)))
	}

	cgocall(asmstdcallAddr, unsafe.Pointer(c))
	handle = c.r1
	if handle == 0 {
		err = c.err
	}
	return
}

//go:linkname syscall_loadlibrary syscall.loadlibrary
//go:nosplit
func syscall_loadlibrary(filename *uint16) (handle, err uintptr) {
	lockOSThread()
	defer unlockOSThread()
	c := &getg().m.syscall
	c.fn = getLoadLibrary()
	c.n = 1
	c.args = uintptr(noescape(unsafe.Pointer(&filename)))
	cgocall(asmstdcallAddr, unsafe.Pointer(c))
	handle = c.r1
	if handle == 0 {
		err = c.err
	}
	return
}

//go:linkname syscall_getprocaddress syscall.getprocaddress
//go:nosplit
func syscall_getprocaddress(handle uintptr, procname *byte) (outhandle, err uintptr) {
	lockOSThread()
	defer unlockOSThread()
	c := &getg().m.syscall
	c.fn = getGetProcAddress()
	c.n = 2
	c.args = uintptr(noescape(unsafe.Pointer(&handle)))
	cgocall(asmstdcallAddr, unsafe.Pointer(c))
	outhandle = c.r1
	if outhandle == 0 {
		err = c.err
	}
	return
}

//go:linkname syscall_Syscall syscall.Syscall
//go:nosplit
func syscall_Syscall(fn, nargs, a1, a2, a3 uintptr) (r1, r2, err uintptr) {
	lockOSThread()
	defer unlockOSThread()
	c := &getg().m.syscall
	c.fn = fn
	c.n = nargs
	c.args = uintptr(noescape(unsafe.Pointer(&a1)))
	cgocall(asmstdcallAddr, unsafe.Pointer(c))
	return c.r1, c.r2, c.err
}

//go:linkname syscall_Syscall6 syscall.Syscall6
//go:nosplit
func syscall_Syscall6(fn, nargs, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr) {
	lockOSThread()
	defer unlockOSThread()
	c := &getg().m.syscall
	c.fn = fn
	c.n = nargs
	c.args = uintptr(noescape(unsafe.Pointer(&a1)))
	cgocall(asmstdcallAddr, unsafe.Pointer(c))
	return c.r1, c.r2, c.err
}

//go:linkname syscall_Syscall9 syscall.Syscall9
//go:nosplit
func syscall_Syscall9(fn, nargs, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r1, r2, err uintptr) {
	lockOSThread()
	defer unlockOSThread()
	c := &getg().m.syscall
	c.fn = fn
	c.n = nargs
	c.args = uintptr(noescape(unsafe.Pointer(&a1)))
	cgocall(asmstdcallAddr, unsafe.Pointer(c))
	return c.r1, c.r2, c.err
}

//go:linkname syscall_Syscall12 syscall.Syscall12
//go:nosplit
func syscall_Syscall12(fn, nargs, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12 uintptr) (r1, r2, err uintptr) {
	lockOSThread()
	defer unlockOSThread()
	c := &getg().m.syscall
	c.fn = fn
	c.n = nargs
	c.args = uintptr(noescape(unsafe.Pointer(&a1)))
	cgocall(asmstdcallAddr, unsafe.Pointer(c))
	return c.r1, c.r2, c.err
}

//go:linkname syscall_Syscall15 syscall.Syscall15
//go:nosplit
func syscall_Syscall15(fn, nargs, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15 uintptr) (r1, r2, err uintptr) {
	lockOSThread()
	defer unlockOSThread()
	c := &getg().m.syscall
	c.fn = fn
	c.n = nargs
	c.args = uintptr(noescape(unsafe.Pointer(&a1)))
	cgocall(asmstdcallAddr, unsafe.Pointer(c))
	return c.r1, c.r2, c.err
}

//go:linkname syscall_Syscall18 syscall.Syscall18
//go:nosplit
func syscall_Syscall18(fn, nargs, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18 uintptr) (r1, r2, err uintptr) {
	lockOSThread()
	defer unlockOSThread()
	c := &getg().m.syscall
	c.fn = fn
	c.n = nargs
	c.args = uintptr(noescape(unsafe.Pointer(&a1)))
	cgocall(asmstdcallAddr, unsafe.Pointer(c))
	return c.r1, c.r2, c.err
}
