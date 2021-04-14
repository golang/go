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
	fn     *funcval // Go function
	retPop uintptr  // For 386 cdecl, how many bytes to pop on return

	// abiMap specifies how to translate from a C frame to a Go
	// frame. This does not specify how to translate back because
	// the result is always a uintptr. If the C ABI is fastcall,
	// this assumes the four fastcall registers were first spilled
	// to the shadow space.
	abiMap []abiPart
	// retOffset is the offset of the uintptr-sized result in the Go
	// frame.
	retOffset uintptr
}

// abiPart encodes a step in translating between calling ABIs.
type abiPart struct {
	src, dst uintptr
	len      uintptr
}

func (a *abiPart) tryMerge(b abiPart) bool {
	if a.src+a.len == b.src && a.dst+a.len == b.dst {
		a.len += b.len
		return true
	}
	return false
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

const callbackMaxFrame = 64 * sys.PtrSize

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

	// Check arguments and construct ABI translation.
	var abiMap []abiPart
	var src, dst uintptr
	for _, t := range ft.in() {
		if t.size > sys.PtrSize {
			// We don't support this right now. In
			// stdcall/cdecl, 64-bit ints and doubles are
			// passed as two words (little endian); and
			// structs are pushed on the stack. In
			// fastcall, arguments larger than the word
			// size are passed by reference. On arm,
			// 8-byte aligned arguments round up to the
			// next even register and can be split across
			// registers and the stack.
			panic("compileCallback: argument size is larger than uintptr")
		}
		if k := t.kind & kindMask; (GOARCH == "amd64" || GOARCH == "arm") && (k == kindFloat32 || k == kindFloat64) {
			// In fastcall, floating-point arguments in
			// the first four positions are passed in
			// floating-point registers, which we don't
			// currently spill. arm passes floating-point
			// arguments in VFP registers, which we also
			// don't support.
			panic("compileCallback: float arguments not supported")
		}

		// The Go ABI aligns arguments.
		dst = alignUp(dst, uintptr(t.align))
		// In the C ABI, we're already on a word boundary.
		// Also, sub-word-sized fastcall register arguments
		// are stored to the least-significant bytes of the
		// argument word and all supported Windows
		// architectures are little endian, so src is already
		// pointing to the right place for smaller arguments.
		// The same is true on arm.

		// Copy just the size of the argument. Note that this
		// could be a small by-value struct, but C and Go
		// struct layouts are compatible, so we can copy these
		// directly, too.
		part := abiPart{src, dst, t.size}
		// Add this step to the adapter.
		if len(abiMap) == 0 || !abiMap[len(abiMap)-1].tryMerge(part) {
			abiMap = append(abiMap, part)
		}

		// cdecl, stdcall, fastcall, and arm pad arguments to word size.
		src += sys.PtrSize
		// The Go ABI packs arguments.
		dst += t.size
	}
	// The Go ABI aligns the result to the word size. src is
	// already aligned.
	dst = alignUp(dst, sys.PtrSize)
	retOffset := dst

	if len(ft.out()) != 1 {
		panic("compileCallback: expected function with one uintptr-sized result")
	}
	if ft.out()[0].size != sys.PtrSize {
		panic("compileCallback: expected function with one uintptr-sized result")
	}
	if k := ft.out()[0].kind & kindMask; k == kindFloat32 || k == kindFloat64 {
		// In cdecl and stdcall, float results are returned in
		// ST(0). In fastcall, they're returned in XMM0.
		// Either way, it's not AX.
		panic("compileCallback: float results not supported")
	}
	// Make room for the uintptr-sized result.
	dst += sys.PtrSize

	if dst > callbackMaxFrame {
		panic("compileCallback: function argument frame too large")
	}

	// For cdecl, the callee is responsible for popping its
	// arguments from the C stack.
	var retPop uintptr
	if cdecl {
		retPop = src
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
	c := winCallback{key.fn, retPop, abiMap, retOffset}
	cbs.ctxt[n] = c
	cbs.index[key] = n
	cbs.n++

	unlock(&cbs.lock)
	return callbackasmAddr(n)
}

type callbackArgs struct {
	index uintptr
	// args points to the argument block.
	//
	// For cdecl and stdcall, all arguments are on the stack.
	//
	// For fastcall, the trampoline spills register arguments to
	// the reserved spill slots below the stack arguments,
	// resulting in a layout equivalent to stdcall.
	//
	// For arm, the trampoline stores the register arguments just
	// below the stack arguments, so again we can treat it as one
	// big stack arguments frame.
	args unsafe.Pointer
	// Below are out-args from callbackWrap
	result uintptr
	retPop uintptr // For 386 cdecl, how many bytes to pop on return
}

// callbackWrap is called by callbackasm to invoke a registered C callback.
func callbackWrap(a *callbackArgs) {
	c := cbs.ctxt[a.index]
	a.retPop = c.retPop

	// Convert from C to Go ABI.
	var frame [callbackMaxFrame]byte
	goArgs := unsafe.Pointer(&frame)
	for _, part := range c.abiMap {
		memmove(add(goArgs, part.dst), add(a.args, part.src), part.len)
	}

	// Even though this is copying back results, we can pass a nil
	// type because those results must not require write barriers.
	reflectcall(nil, unsafe.Pointer(c.fn), noescape(goArgs), uint32(c.retOffset)+sys.PtrSize, uint32(c.retOffset))

	// Extract the result.
	a.result = *(*uintptr)(unsafe.Pointer(&frame[c.retOffset]))
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
