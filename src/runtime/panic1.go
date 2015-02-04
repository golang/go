// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// Code related to defer, panic and recover.
// TODO: Merge into panic.go.

//uint32 runtime·panicking;
var paniclk mutex

const hasLinkRegister = GOARCH == "arm" || GOARCH == "ppc64" || GOARCH == "ppc64le"

// Unwind the stack after a deferred function calls recover
// after a panic.  Then arrange to continue running as though
// the caller of the deferred function returned normally.
func recovery(gp *g) {
	// Info about defer passed in G struct.
	sp := gp.sigcode0
	pc := gp.sigcode1

	// d's arguments need to be in the stack.
	if sp != 0 && (sp < gp.stack.lo || gp.stack.hi < sp) {
		print("recover: ", hex(sp), " not in [", hex(gp.stack.lo), ", ", hex(gp.stack.hi), "]\n")
		throw("bad recovery")
	}

	// Make the deferproc for this d return again,
	// this time returning 1.  The calling function will
	// jump to the standard return epilogue.
	gp.sched.sp = sp
	gp.sched.pc = pc
	gp.sched.lr = 0
	gp.sched.ret = 1
	gogo(&gp.sched)
}

func startpanic_m() {
	_g_ := getg()
	if mheap_.cachealloc.size == 0 { // very early
		print("runtime: panic before malloc heap initialized\n")
		_g_.m.mallocing = 1 // tell rest of panic not to try to malloc
	} else if _g_.m.mcache == nil { // can happen if called from signal handler or throw
		_g_.m.mcache = allocmcache()
	}

	switch _g_.m.dying {
	case 0:
		_g_.m.dying = 1
		if _g_ != nil {
			_g_.writebuf = nil
		}
		xadd(&panicking, 1)
		lock(&paniclk)
		if debug.schedtrace > 0 || debug.scheddetail > 0 {
			schedtrace(true)
		}
		freezetheworld()
		return
	case 1:
		// Something failed while panicing, probably the print of the
		// argument to panic().  Just print a stack trace and exit.
		_g_.m.dying = 2
		print("panic during panic\n")
		dopanic(0)
		exit(3)
		fallthrough
	case 2:
		// This is a genuine bug in the runtime, we couldn't even
		// print the stack trace successfully.
		_g_.m.dying = 3
		print("stack trace unavailable\n")
		exit(4)
		fallthrough
	default:
		// Can't even print!  Just exit.
		exit(5)
	}
}

var didothers bool
var deadlock mutex

func dopanic_m(gp *g, pc, sp uintptr) {
	if gp.sig != 0 {
		print("[signal ", hex(gp.sig), " code=", hex(gp.sigcode0), " addr=", hex(gp.sigcode1), " pc=", hex(gp.sigpc), "]\n")
	}

	var docrash bool
	_g_ := getg()
	if t := gotraceback(&docrash); t > 0 {
		if gp != gp.m.g0 {
			print("\n")
			goroutineheader(gp)
			traceback(pc, sp, 0, gp)
		} else if t >= 2 || _g_.m.throwing > 0 {
			print("\nruntime stack:\n")
			traceback(pc, sp, 0, gp)
		}
		if !didothers {
			didothers = true
			tracebackothers(gp)
		}
	}
	unlock(&paniclk)

	if xadd(&panicking, -1) != 0 {
		// Some other m is panicking too.
		// Let it print what it needs to print.
		// Wait forever without chewing up cpu.
		// It will exit when it's done.
		lock(&deadlock)
		lock(&deadlock)
	}

	if docrash {
		crash()
	}

	exit(2)
}

//go:nosplit
func canpanic(gp *g) bool {
	// Note that g is m->gsignal, different from gp.
	// Note also that g->m can change at preemption, so m can go stale
	// if this function ever makes a function call.
	_g_ := getg()
	_m_ := _g_.m

	// Is it okay for gp to panic instead of crashing the program?
	// Yes, as long as it is running Go code, not runtime code,
	// and not stuck in a system call.
	if gp == nil || gp != _m_.curg {
		return false
	}
	if _m_.locks-_m_.softfloat != 0 || _m_.mallocing != 0 || _m_.throwing != 0 || _m_.preemptoff != "" || _m_.dying != 0 {
		return false
	}
	status := readgstatus(gp)
	if status&^_Gscan != _Grunning || gp.syscallsp != 0 {
		return false
	}
	if GOOS == "windows" && _m_.libcallsp != 0 {
		return false
	}
	return true
}
