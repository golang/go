// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"unsafe"
)

type callbacks struct {
	lock
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

func compileCallback(fn eface, cleanstack bool) (code uintptr) {
	if fn._type == nil || (fn._type.kind&kindMask) != kindFunc {
		panic("compilecallback: not a function")
	}
	ft := (*functype)(unsafe.Pointer(fn._type))
	if len(ft.out) != 1 {
		panic("compilecallback: function must have one output parameter")
	}
	uintptrSize := unsafe.Sizeof(uintptr(0))
	if t := (**_type)(unsafe.Pointer(&ft.out[0])); (*t).size != uintptrSize {
		panic("compilecallback: output parameter size is wrong")
	}
	argsize := uintptr(0)
	for _, t := range (*[1024](*_type))(unsafe.Pointer(&ft.in[0]))[:len(ft.in)] {
		if (*t).size != uintptrSize {
			panic("compilecallback: input parameter size is wrong")
		}
		argsize += uintptrSize
	}

	golock(&cbs.lock)
	defer gounlock(&cbs.lock)

	n := cbs.n
	for i := 0; i < n; i++ {
		if cbs.ctxt[i].gobody == fn.data && cbs.ctxt[i].isCleanstack() == cleanstack {
			return callbackasmAddr(i)
		}
	}
	if n >= cb_max {
		gothrow("too many callback functions")
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
