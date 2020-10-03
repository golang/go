// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/sys"
	"unsafe"
)

// cbs stores all registered Go callbacks.
var cbs struct {
	lock  mutex
	ctxt  [cb_max]winCallback
	index map[winCallbackKey]int
	n     int
}

// winCallback records information about a registered Go callback.
type winCallback struct {
	fn      *funcval // Go function
	argsize uintptr  // Callback arguments size (in bytes)
	cdecl   bool     // C function uses cdecl calling convention
}

type winCallbackKey struct {
	fn    *funcval
	cdecl bool
}

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

const callbackMaxArgs = 64

// compileCallback converts a Go function fn into a C function pointer
// that can be passed to Windows APIs.
//
// On 386, if cdecl is true, the returned C function will use the
// cdecl calling convention; otherwise, it will use stdcall. On amd64,
// it always uses fastcall. On arm, it always uses the ARM convention.
//
//go:linkname compileCallback syscall.compileCallback
func compileCallback(fn eface, cdecl bool) (code uintptr) {
	if GOARCH != "386" {
		// cdecl is only meaningful on 386.
		cdecl = false
	}

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
	if len(ft.in()) > callbackMaxArgs {
		panic("compileCallback: too many function arguments")
	}
	argsize := uintptr(0)
	for _, t := range ft.in() {
		if t.size > uintptrSize {
			panic("compileCallback: argument size is larger than uintptr")
		}
		argsize += uintptrSize
	}

	key := winCallbackKey{(*funcval)(fn.data), cdecl}

	lock(&cbs.lock) // We don't unlock this in a defer because this is used from the system stack.

	// Check if this callback is already registered.
	if n, ok := cbs.index[key]; ok {
		unlock(&cbs.lock)
		return callbackasmAddr(n)
	}

	// Register the callback.
	if cbs.index == nil {
		cbs.index = make(map[winCallbackKey]int)
	}
	n := cbs.n
	if n >= len(cbs.ctxt) {
		unlock(&cbs.lock)
		throw("too many callback functions")
	}
	c := winCallback{key.fn, argsize, cdecl}
	cbs.ctxt[n] = c
	cbs.index[key] = n
	cbs.n++

	unlock(&cbs.lock)
	return callbackasmAddr(n)
}

type callbackArgs struct {
	index uintptr
	args  *uintptr // Arguments in stdcall/cdecl convention, with registers spilled
	// Below are out-args from callbackWrap
	result uintptr
	retPop uintptr // For 386 cdecl, how many bytes to pop on return
}

// callbackWrap is called by callbackasm to invoke a registered C callback.
func callbackWrap(a *callbackArgs) {
	c := cbs.ctxt[a.index]
	if GOARCH == "386" {
		if c.cdecl {
			// In cdecl, the callee is responsible for
			// popping its arguments.
			a.retPop = c.argsize
		} else {
			a.retPop = 0
		}
	}

	// Convert from stdcall to Go ABI. We assume the stack layout
	// is the same, and we just need to make room for the result.
	//
	// TODO: This isn't a good assumption. For example, a function
	// that takes two uint16 arguments will be laid out
	// differently by the stdcall and Go ABIs. We should implement
	// proper ABI conversion.
	var frame [callbackMaxArgs + 1]uintptr
	memmove(unsafe.Pointer(&frame), unsafe.Pointer(a.args), c.argsize)

	// Even though this is copying back results, we can pass a nil
	// type because those results must not require write barriers.
	reflectcall(nil, unsafe.Pointer(c.fn), noescape(unsafe.Pointer(&frame)), sys.PtrSize+uint32(c.argsize), uint32(c.argsize))

	// Extract the result.
	a.result = frame[c.argsize/sys.PtrSize]
}

const _LOAD_LIBRARY_SEARCH_SYSTEM32 = 0x00000800

// When available, this function will use LoadLibraryEx with the filename
// parameter and the important SEARCH_SYSTEM32 argument. But on systems that
// do not have that option, absoluteFilepath should contain a fallback
// to the full path inside of system32 for use with vanilla LoadLibrary.
//go:linkname syscall_loadsystemlibrary syscall.loadsystemlibrary
//go:nosplit
func syscall_loadsystemlibrary(filename *uint16, absoluteFilepath *uint16) (handle, err uintptr) {
	lockOSThread()
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
		c.fn = getLoadLibrary()
		c.n = 1
		c.args = uintptr(noescape(unsafe.Pointer(&absoluteFilepath)))
	}

	cgocall(asmstdcallAddr, unsafe.Pointer(c))
	handle = c.r1
	if handle == 0 {
		err = c.err
	}
	unlockOSThread() // not defer'd after the lockOSThread above to save stack frame size.
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
