// Copyright 2014 The Go Authors.  All rights reserved.
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

	callbackasm byte // type isn't really byte, it's code in runtime
)

// callbackasmAddr returns address of runtime.callbackasm
// function adjusted by i.
// runtime.callbackasm is just a series of CALL instructions
// (each is 5 bytes long), and we want callback to arrive at
// correspondent call instruction instead of start of
// runtime.callbackasm.
func callbackasmAddr(i int) uintptr {
	return uintptr(add(unsafe.Pointer(&callbackasm), uintptr(i*5)))
}

//go:linkname compileCallback syscall.compileCallback
func compileCallback(fn eface, cleanstack bool) (code uintptr) {
	if fn._type == nil || (fn._type.kind&kindMask) != kindFunc {
		panic("compileCallback: not a function")
	}
	ft := (*functype)(unsafe.Pointer(fn._type))
	if ft.out.len != 1 {
		panic("compileCallback: function must have one output parameter")
	}
	uintptrSize := unsafe.Sizeof(uintptr(0))
	if t := (**_type)(unsafe.Pointer(ft.out.array)); (*t).size != uintptrSize {
		panic("compileCallback: output parameter size is wrong")
	}
	argsize := uintptr(0)
	for _, t := range (*[1024](*_type))(unsafe.Pointer(ft.in.array))[:ft.in.len] {
		if (*t).size > uintptrSize {
			panic("compileCallback: input parameter size is wrong")
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

//go:linkname syscall_loadlibrary syscall.loadlibrary
//go:nosplit
func syscall_loadlibrary(filename *uint16) (handle, err uintptr) {
	var c libcall
	c.fn = getLoadLibrary()
	c.n = 1
	c.args = uintptr(unsafe.Pointer(&filename))
	cgocall_errno(unsafe.Pointer(funcPC(asmstdcall)), unsafe.Pointer(&c))
	handle = c.r1
	if handle == 0 {
		err = c.err
	}
	return
}

//go:linkname syscall_getprocaddress syscall.getprocaddress
//go:nosplit
func syscall_getprocaddress(handle uintptr, procname *byte) (outhandle, err uintptr) {
	var c libcall
	c.fn = getGetProcAddress()
	c.n = 2
	c.args = uintptr(unsafe.Pointer(&handle))
	cgocall_errno(unsafe.Pointer(funcPC(asmstdcall)), unsafe.Pointer(&c))
	outhandle = c.r1
	if outhandle == 0 {
		err = c.err
	}
	return
}

//go:linkname syscall_Syscall syscall.Syscall
//go:nosplit
func syscall_Syscall(fn, nargs, a1, a2, a3 uintptr) (r1, r2, err uintptr) {
	var c libcall
	c.fn = fn
	c.n = nargs
	c.args = uintptr(unsafe.Pointer(&a1))
	cgocall_errno(unsafe.Pointer(funcPC(asmstdcall)), unsafe.Pointer(&c))
	return c.r1, c.r2, c.err
}

//go:linkname syscall_Syscall6 syscall.Syscall6
//go:nosplit
func syscall_Syscall6(fn, nargs, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr) {
	var c libcall
	c.fn = fn
	c.n = nargs
	c.args = uintptr(unsafe.Pointer(&a1))
	cgocall_errno(unsafe.Pointer(funcPC(asmstdcall)), unsafe.Pointer(&c))
	return c.r1, c.r2, c.err
}

//go:linkname syscall_Syscall9 syscall.Syscall9
//go:nosplit
func syscall_Syscall9(fn, nargs, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r1, r2, err uintptr) {
	var c libcall
	c.fn = fn
	c.n = nargs
	c.args = uintptr(unsafe.Pointer(&a1))
	cgocall_errno(unsafe.Pointer(funcPC(asmstdcall)), unsafe.Pointer(&c))
	return c.r1, c.r2, c.err
}

//go:linkname syscall_Syscall12 syscall.Syscall12
//go:nosplit
func syscall_Syscall12(fn, nargs, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12 uintptr) (r1, r2, err uintptr) {
	var c libcall
	c.fn = fn
	c.n = nargs
	c.args = uintptr(unsafe.Pointer(&a1))
	cgocall_errno(unsafe.Pointer(funcPC(asmstdcall)), unsafe.Pointer(&c))
	return c.r1, c.r2, c.err
}

//go:linkname syscall_Syscall15 syscall.Syscall15
//go:nosplit
func syscall_Syscall15(fn, nargs, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15 uintptr) (r1, r2, err uintptr) {
	var c libcall
	c.fn = fn
	c.n = nargs
	c.args = uintptr(unsafe.Pointer(&a1))
	cgocall_errno(unsafe.Pointer(funcPC(asmstdcall)), unsafe.Pointer(&c))
	return c.r1, c.r2, c.err
}
