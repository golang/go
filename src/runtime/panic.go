// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

var indexError = error(errorString("index out of range"))

func panicindex() {
	panic(indexError)
}

var sliceError = error(errorString("slice bounds out of range"))

func panicslice() {
	panic(sliceError)
}

var divideError = error(errorString("integer divide by zero"))

func panicdivide() {
	panic(divideError)
}

var overflowError = error(errorString("integer overflow"))

func panicoverflow() {
	panic(overflowError)
}

var floatError = error(errorString("floating point error"))

func panicfloat() {
	panic(floatError)
}

var memoryError = error(errorString("invalid memory address or nil pointer dereference"))

func panicmem() {
	panic(memoryError)
}

func throwreturn() {
	gothrow("no return at end of a typed function - compiler is broken")
}

func throwinit() {
	gothrow("recursive call during initialization - linker skew")
}

// Create a new deferred function fn with siz bytes of arguments.
// The compiler turns a defer statement into a call to this.
//go:nosplit
func deferproc(siz int32, fn *funcval) { // arguments of fn follow fn
	// the arguments of fn are in a perilous state.  The stack map
	// for deferproc does not describe them.  So we can't let garbage
	// collection or stack copying trigger until we've copied them out
	// to somewhere safe.  deferproc_m does that.  Until deferproc_m,
	// we can only call nosplit routines.
	argp := uintptr(unsafe.Pointer(&fn))
	argp += unsafe.Sizeof(fn)
	if GOARCH == "arm" {
		argp += ptrSize // skip caller's saved link register
	}
	mp := acquirem()
	mp.scalararg[0] = uintptr(siz)
	mp.ptrarg[0] = unsafe.Pointer(fn)
	mp.scalararg[1] = argp
	mp.scalararg[2] = getcallerpc(unsafe.Pointer(&siz))

	if mp.curg != getg() {
		// go code on the m stack can't defer
		gothrow("defer on m")
	}

	onM(deferproc_m)

	releasem(mp)

	// deferproc returns 0 normally.
	// a deferred func that stops a panic
	// makes the deferproc return 1.
	// the code the compiler generates always
	// checks the return value and jumps to the
	// end of the function if deferproc returns != 0.
	return0()
	// No code can go here - the C return register has
	// been set and must not be clobbered.
}

// Each P holds pool for defers with arg sizes 8, 24, 40, 56 and 72 bytes.
// Memory block is 40 (24 for 32 bits) bytes larger due to Defer header.
// This maps exactly to malloc size classes.

// defer size class for arg size sz
func deferclass(siz uintptr) uintptr {
	return (siz + 7) >> 4
}

// total size of memory block for defer with arg size sz
func totaldefersize(siz uintptr) uintptr {
	return (unsafe.Sizeof(_defer{}) - unsafe.Sizeof(_defer{}.args)) + round(siz, ptrSize)
}

// Ensure that defer arg sizes that map to the same defer size class
// also map to the same malloc size class.
func testdefersizes() {
	var m [len(p{}.deferpool)]int32

	for i := range m {
		m[i] = -1
	}
	for i := uintptr(0); ; i++ {
		defersc := deferclass(i)
		if defersc >= uintptr(len(m)) {
			break
		}
		siz := goroundupsize(totaldefersize(i))
		if m[defersc] < 0 {
			m[defersc] = int32(siz)
			continue
		}
		if m[defersc] != int32(siz) {
			print("bad defer size class: i=", i, " siz=", siz, " defersc=", defersc, "\n")
			gothrow("bad defer size class")
		}
	}
}

// Allocate a Defer, usually using per-P pool.
// Each defer must be released with freedefer.
// Note: runs on M stack
func newdefer(siz int32) *_defer {
	var d *_defer
	sc := deferclass(uintptr(siz))
	mp := acquirem()
	if sc < uintptr(len(p{}.deferpool)) {
		pp := mp.p
		d = pp.deferpool[sc]
		if d != nil {
			pp.deferpool[sc] = d.link
		}
	}
	if d == nil {
		// deferpool is empty or just a big defer
		total := goroundupsize(totaldefersize(uintptr(siz)))
		d = (*_defer)(gomallocgc(total, conservative, 0))
	}
	d.siz = siz
	d.special = false
	gp := mp.curg
	d.link = gp._defer
	gp._defer = d
	releasem(mp)
	return d
}

// Free the given defer.
// The defer cannot be used after this call.
func freedefer(d *_defer) {
	if d.special {
		return
	}
	sc := deferclass(uintptr(d.siz))
	if sc < uintptr(len(p{}.deferpool)) {
		mp := acquirem()
		pp := mp.p
		d.link = pp.deferpool[sc]
		pp.deferpool[sc] = d
		releasem(mp)
		// No need to wipe out pointers in argp/pc/fn/args,
		// because we empty the pool before GC.
	}
}

// Run a deferred function if there is one.
// The compiler inserts a call to this at the end of any
// function which calls defer.
// If there is a deferred function, this will call runtime·jmpdefer,
// which will jump to the deferred function such that it appears
// to have been called by the caller of deferreturn at the point
// just before deferreturn was called.  The effect is that deferreturn
// is called again and again until there are no more deferred functions.
// Cannot split the stack because we reuse the caller's frame to
// call the deferred function.

// The single argument isn't actually used - it just has its address
// taken so it can be matched against pending defers.
//go:nosplit
func deferreturn(arg0 uintptr) {
	gp := getg()
	d := gp._defer
	if d == nil {
		return
	}
	argp := uintptr(unsafe.Pointer(&arg0))
	if d.argp != argp {
		return
	}

	// Moving arguments around.
	// Do not allow preemption here, because the garbage collector
	// won't know the form of the arguments until the jmpdefer can
	// flip the PC over to fn.
	mp := acquirem()
	memmove(unsafe.Pointer(argp), unsafe.Pointer(&d.args), uintptr(d.siz))
	fn := d.fn
	gp._defer = d.link
	freedefer(d)
	releasem(mp)
	jmpdefer(fn, argp)
}

// Goexit terminates the goroutine that calls it.  No other goroutine is affected.
// Goexit runs all deferred calls before terminating the goroutine.
//
// Calling Goexit from the main goroutine terminates that goroutine
// without func main returning. Since func main has not returned,
// the program continues execution of other goroutines.
// If all other goroutines exit, the program crashes.
func Goexit() {
	// Run all deferred functions for the current goroutine.
	gp := getg()
	for gp._defer != nil {
		d := gp._defer
		gp._defer = d.link
		reflectcall(unsafe.Pointer(d.fn), unsafe.Pointer(&d.args), uint32(d.siz), uint32(d.siz))
		freedefer(d)
		// Note: we ignore recovers here because Goexit isn't a panic
	}
	goexit()
}

func canpanic(*g) bool

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
		argp := unsafe.Pointer(d.argp) // must be pointer so it gets adjusted during stack copy
		pc := d.pc

		// The deferred function may cause another panic,
		// so reflectcall may not return. Set up a defer
		// to mark this panic aborted if that happens.
		dabort.link = gp._defer
		gp._defer = (*_defer)(noescape(unsafe.Pointer(&dabort)))
		p._defer = d

		p.argp = getargp(0)
		reflectcall(unsafe.Pointer(d.fn), unsafe.Pointer(&d.args), uint32(d.siz), uint32(d.siz))
		p.argp = 0

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

// getargp returns the location where the caller
// writes outgoing function call arguments.
//go:nosplit
func getargp(x int) uintptr {
	// x is an argument mainly so that we can return its address.
	// However, we need to make the function complex enough
	// that it won't be inlined. We always pass x = 0, so this code
	// does nothing other than keep the compiler from thinking
	// the function is simple enough to inline.
	if x > 0 {
		return getcallersp(unsafe.Pointer(&x)) * 0
	}
	return uintptr(noescape(unsafe.Pointer(&x)))
}

func abortpanic(p *_panic) {
	p.aborted = true
}

// The implementation of the predeclared function recover.
// Cannot split the stack because it needs to reliably
// find the stack segment of its caller.
//
// TODO(rsc): Once we commit to CopyStackAlways,
// this doesn't need to be nosplit.
//go:nosplit
func gorecover(argp uintptr) interface{} {
	// Must be in a function running as part of a deferred call during the panic.
	// Must be called from the topmost function of the call
	// (the function used in the defer statement).
	// p.argp is the argument pointer of that topmost deferred function call.
	// Compare against argp reported by caller.
	// If they match, the caller is the one who can recover.
	gp := getg()
	p := gp._panic
	if p != nil && !p.recovered && argp == p.argp {
		p.recovered = true
		return p.arg
	}
	return nil
}

//go:nosplit
func startpanic() {
	onM(startpanic_m)
}

//go:nosplit
func dopanic(unused int) {
	gp := getg()
	mp := acquirem()
	mp.ptrarg[0] = unsafe.Pointer(gp)
	mp.scalararg[0] = getcallerpc((unsafe.Pointer)(&unused))
	mp.scalararg[1] = getcallersp((unsafe.Pointer)(&unused))
	onM(dopanic_m) // should never return
	*(*int)(nil) = 0
}

//go:nosplit
func throw(s *byte) {
	gp := getg()
	if gp.m.throwing == 0 {
		gp.m.throwing = 1
	}
	startpanic()
	print("fatal error: ", gostringnocopy(s), "\n")
	dopanic(0)
	*(*int)(nil) = 0 // not reached
}

//go:nosplit
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
