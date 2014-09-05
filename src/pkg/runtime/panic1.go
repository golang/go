// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// Print all currently active panics.  Used when crashing.
func printpanics(p *_panic) {
	if p.link != nil {
		printpanics(p.link)
		print("\t")
	}
	print("panic: ")
	printany(p.arg)
	if p.recovered {
		print(" [recovered]")
	}
	print("\n")
}

// The implementation of the predeclared function panic.
func gopanic(e interface{}) {
	gp := getg()
	if gp.m.curg != gp {
		gothrow("panic on m stack")
	}
	var p _panic
	var dabort _defer
	p.arg = e
	p.link = gp._panic
	gp._panic = (*_panic)(noescape(unsafe.Pointer(&p)))

	fn := abortpanic
	dabort.fn = *(**funcval)(unsafe.Pointer(&fn))
	dabort.siz = ptrSize
	dabort.args[0] = noescape((unsafe.Pointer)(&p)) // TODO(khr): why do I need noescape here?
	dabort.argp = _NoArgs
	dabort.special = true

	for {
		d := gp._defer
		if d == nil {
			break
		}
		// take defer off list in case of recursive panic
		gp._defer = d.link
		gp.ispanic = true              // rock for runtime·newstack, where runtime·newstackcall ends up
		argp := unsafe.Pointer(d.argp) // must be pointer so it gets adjusted during stack copy
		pc := d.pc

		// The deferred function may cause another panic,
		// so newstackcall may not return. Set up a defer
		// to mark this panic aborted if that happens.
		dabort.link = gp._defer
		gp._defer = (*_defer)(noescape(unsafe.Pointer(&dabort)))
		p._defer = d
		p.outerwrap = gp.panicwrap

		// TODO(rsc): I am pretty sure the panicwrap manipulation here is not correct.
		// It is close enough to pass all the tests we have, but I think it needs to be
		// restored during recovery too. I will write better tests and fix it in a separate CL.

		gp.panicwrap = 0
		reflectcall(unsafe.Pointer(d.fn), unsafe.Pointer(&d.args), uint32(d.siz), uint32(d.siz), (*_panic)(noescape(unsafe.Pointer(&p))))
		gp.panicwrap = p.outerwrap

		// reflectcall did not panic. Remove dabort.
		if gp._defer != &dabort {
			gothrow("bad defer entry in panic")
		}
		gp._defer = dabort.link

		// trigger shrinkage to test stack copy.  See stack_test.go:TestStackPanic
		//GC()

		freedefer(d)
		if p.recovered {
			gp._panic = p.link
			// Aborted panics are marked but remain on the g.panic list.
			// Remove them from the list and free the associated defers.
			for gp._panic != nil && gp._panic.aborted {
				freedefer(gp._panic._defer)
				gp._panic = gp._panic.link
			}
			if gp._panic == nil { // must be done with signal
				gp.sig = 0
			}
			// Pass information about recovering frame to recovery.
			gp.sigcode0 = uintptr(argp)
			gp.sigcode1 = pc
			mcall(recovery_m)
			gothrow("recovery failed") // mcall should not return
		}
	}

	// ran out of deferred calls - old-school panic now
	startpanic()
	printpanics(gp._panic)
	dopanic(0)       // should not return
	*(*int)(nil) = 0 // not reached
}

func abortpanic(p *_panic) {
	p.aborted = true
}

// The implementation of the predeclared function recover.
// Cannot split the stack because it needs to reliably
// find the stack segment of its caller.
//go:nosplit
func gorecover(argp uintptr) interface{} {
	// Must be an unrecovered panic in progress.
	// Must be on a stack segment created for a deferred call during a panic.
	// Must be at the top of that segment, meaning the deferred call itself
	// and not something it called. The top frame in the segment will have
	// argument pointer argp == top - top.argsize.
	// The subtraction of g.panicwrap allows wrapper functions that
	// do not count as official calls to adjust what we consider the top frame
	// while they are active on the stack. The linker emits adjustments of
	// g.panicwrap in the prologue and epilogue of functions marked as wrappers.
	gp := getg()
	p := gp._panic
	//	if p != nil {
	//		println("recover?", p, p.recovered, hex(argp), hex(p.argp), uintptr(gp.panicwrap), p != nil && !p.recovered && argp == p.argp-uintptr(gp.panicwrap))
	//	}
	if p != nil && !p.recovered && argp == p.argp-uintptr(gp.panicwrap) {
		p.recovered = true
		return p.arg
	}
	return nil
}

func startpanic() {
	onM(startpanic_m)
}

func dopanic(unused int) {
	gp := getg()
	mp := acquirem()
	mp.ptrarg[0] = unsafe.Pointer(gp)
	mp.scalararg[0] = getcallerpc((unsafe.Pointer)(&unused))
	mp.scalararg[1] = getcallersp((unsafe.Pointer)(&unused))
	onM(dopanic_m) // should never return
	*(*int)(nil) = 0
}

func throw(s *byte) {
	gothrow(gostringnocopy(s))
}

func gothrow(s string) {
	gp := getg()
	if gp.m.throwing == 0 {
		gp.m.throwing = 1
	}
	startpanic()
	print("fatal error: ", s, "\n")
	dopanic(0)
	*(*int)(nil) = 0 // not reached
}

func panicstring(s *int8) {
	// m.softfloat is set during software floating point,
	// which might cause a fault during a memory load.
	// It increments m.locks to avoid preemption.
	// If we're panicking, the software floating point frames
	// will be unwound, so decrement m.locks as they would.
	gp := getg()
	if gp.m.softfloat != 0 {
		gp.m.locks--
		gp.m.softfloat = 0
	}

	if gp.m.mallocing != 0 {
		print("panic: ", s, "\n")
		gothrow("panic during malloc")
	}
	if gp.m.gcing != 0 {
		print("panic: ", s, "\n")
		gothrow("panic during gc")
	}
	if gp.m.locks != 0 {
		print("panic: ", s, "\n")
		gothrow("panic holding locks")
	}

	var err interface{}
	newErrorCString(unsafe.Pointer(s), &err)
	gopanic(err)
}
