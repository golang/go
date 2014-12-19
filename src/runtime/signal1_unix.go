// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package runtime

const (
	_SIG_DFL uintptr = 0
	_SIG_IGN uintptr = 1
)

func initsig() {
	// _NSIG is the number of signals on this operating system.
	// sigtable should describe what to do for all the possible signals.
	if len(sigtable) != _NSIG {
		print("runtime: len(sigtable)=", len(sigtable), " _NSIG=", _NSIG, "\n")
		gothrow("initsig")
	}

	// First call: basic setup.
	for i := int32(0); i < _NSIG; i++ {
		t := &sigtable[i]
		if t.flags == 0 || t.flags&_SigDefault != 0 {
			continue
		}

		// For some signals, we respect an inherited SIG_IGN handler
		// rather than insist on installing our own default handler.
		// Even these signals can be fetched using the os/signal package.
		switch i {
		case _SIGHUP, _SIGINT:
			if getsig(i) == _SIG_IGN {
				t.flags = _SigNotify | _SigIgnored
				continue
			}
		}

		if t.flags&_SigSetStack != 0 {
			setsigstack(i)
			continue
		}

		t.flags |= _SigHandling
		setsig(i, funcPC(sighandler), true)
	}
}

func sigenable(sig uint32) {
	if sig >= uint32(len(sigtable)) {
		return
	}

	t := &sigtable[sig]
	if t.flags&_SigNotify != 0 && t.flags&_SigHandling == 0 {
		t.flags |= _SigHandling
		if getsig(int32(sig)) == _SIG_IGN {
			t.flags |= _SigIgnored
		}
		setsig(int32(sig), funcPC(sighandler), true)
	}
}

func sigdisable(sig uint32) {
	if sig >= uint32(len(sigtable)) {
		return
	}

	t := &sigtable[sig]
	if t.flags&_SigNotify != 0 && t.flags&_SigHandling != 0 {
		t.flags &^= _SigHandling
		if t.flags&_SigIgnored != 0 {
			setsig(int32(sig), _SIG_IGN, true)
		} else {
			setsig(int32(sig), _SIG_DFL, true)
		}
	}
}

func resetcpuprofiler(hz int32) {
	var it itimerval
	if hz == 0 {
		setitimer(_ITIMER_PROF, &it, nil)
	} else {
		it.it_interval.tv_sec = 0
		it.it_interval.set_usec(1000000 / hz)
		it.it_value = it.it_interval
		setitimer(_ITIMER_PROF, &it, nil)
	}
	_g_ := getg()
	_g_.m.profilehz = hz
}

func sigpipe() {
	setsig(_SIGPIPE, _SIG_DFL, false)
	raise(_SIGPIPE)
}

func crash() {
	if GOOS == "darwin" {
		// OS X core dumps are linear dumps of the mapped memory,
		// from the first virtual byte to the last, with zeros in the gaps.
		// Because of the way we arrange the address space on 64-bit systems,
		// this means the OS X core file will be >128 GB and even on a zippy
		// workstation can take OS X well over an hour to write (uninterruptible).
		// Save users from making that mistake.
		if ptrSize == 8 {
			return
		}
	}

	unblocksignals()
	setsig(_SIGABRT, _SIG_DFL, false)
	raise(_SIGABRT)
}
