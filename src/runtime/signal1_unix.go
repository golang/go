// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package runtime

const (
	_SIG_DFL uintptr = 0
	_SIG_IGN uintptr = 1
)

// Stores the signal handlers registered before Go installed its own.
// These signal handlers will be invoked in cases where Go doesn't want to
// handle a particular signal (e.g., signal occurred on a non-Go thread).
// See sigfwdgo() for more information on when the signals are forwarded.
//
// Signal forwarding is currently available only on Linux.
var fwdSig [_NSIG]uintptr

// sigmask represents a general signal mask compatible with the GOOS
// specific sigset types: the signal numbered x is represented by bit x-1
// to match the representation expected by sigprocmask.
type sigmask [(_NSIG + 31) / 32]uint32

// channels for synchronizing signal mask updates with the signal mask
// thread
var (
	disableSigChan  chan uint32
	enableSigChan   chan uint32
	maskUpdatedChan chan struct{}
)

func initsig() {
	// _NSIG is the number of signals on this operating system.
	// sigtable should describe what to do for all the possible signals.
	if len(sigtable) != _NSIG {
		print("runtime: len(sigtable)=", len(sigtable), " _NSIG=", _NSIG, "\n")
		throw("initsig")
	}

	// First call: basic setup.
	for i := int32(0); i < _NSIG; i++ {
		t := &sigtable[i]
		if t.flags == 0 || t.flags&_SigDefault != 0 {
			continue
		}
		fwdSig[i] = getsig(i)
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
	if t.flags&_SigNotify != 0 {
		ensureSigM()
		enableSigChan <- sig
		<-maskUpdatedChan
		if t.flags&_SigHandling == 0 {
			t.flags |= _SigHandling
			if getsig(int32(sig)) == _SIG_IGN {
				t.flags |= _SigIgnored
			}
			setsig(int32(sig), funcPC(sighandler), true)
		}
	}
}

func sigdisable(sig uint32) {
	if sig >= uint32(len(sigtable)) {
		return
	}

	t := &sigtable[sig]
	if t.flags&_SigNotify != 0 {
		ensureSigM()
		disableSigChan <- sig
		<-maskUpdatedChan
		if t.flags&_SigHandling != 0 {
			t.flags &^= _SigHandling
			if t.flags&_SigIgnored != 0 {
				setsig(int32(sig), _SIG_IGN, true)
			} else {
				setsig(int32(sig), _SIG_DFL, true)
			}
		}
	}
}

func sigignore(sig uint32) {
	if sig >= uint32(len(sigtable)) {
		return
	}

	t := &sigtable[sig]
	if t.flags&_SigNotify != 0 {
		t.flags &^= _SigHandling
		setsig(int32(sig), _SIG_IGN, true)
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

// raisebadsignal is called when a signal is received on a non-Go
// thread, and the Go program does not want to handle it (that is, the
// program has not called os/signal.Notify for the signal).
func raisebadsignal(sig int32) {
	if sig == _SIGPROF {
		// Ignore profiling signals that arrive on non-Go threads.
		return
	}

	var handler uintptr
	if sig >= _NSIG {
		handler = _SIG_DFL
	} else {
		handler = fwdSig[sig]
	}

	// Reset the signal handler and raise the signal.
	// We are currently running inside a signal handler, so the
	// signal is blocked.  We need to unblock it before raising the
	// signal, or the signal we raise will be ignored until we return
	// from the signal handler.  We know that the signal was unblocked
	// before entering the handler, or else we would not have received
	// it.  That means that we don't have to worry about blocking it
	// again.
	unblocksig(sig)
	setsig(sig, handler, false)
	raise(sig)

	// If the signal didn't cause the program to exit, restore the
	// Go signal handler and carry on.
	//
	// We may receive another instance of the signal before we
	// restore the Go handler, but that is not so bad: we know
	// that the Go program has been ignoring the signal.
	setsig(sig, funcPC(sighandler), true)
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

	updatesigmask(sigmask{})
	setsig(_SIGABRT, _SIG_DFL, false)
	raise(_SIGABRT)
}

// createSigM starts one global, sleeping thread to make sure at least one thread
// is available to catch signals enabled for os/signal.
func ensureSigM() {
	if maskUpdatedChan != nil {
		return
	}
	maskUpdatedChan = make(chan struct{})
	disableSigChan = make(chan uint32)
	enableSigChan = make(chan uint32)
	go func() {
		// Signal masks are per-thread, so make sure this goroutine stays on one
		// thread.
		LockOSThread()
		defer UnlockOSThread()
		// The sigBlocked mask contains the signals not active for os/signal,
		// initially all signals except the essential. When signal.Notify()/Stop is called,
		// sigenable/sigdisable in turn notify this thread to update its signal
		// mask accordingly.
		var sigBlocked sigmask
		for i := range sigBlocked {
			sigBlocked[i] = ^uint32(0)
		}
		for i := range sigtable {
			if sigtable[i].flags&_SigUnblock != 0 {
				sigBlocked[(i-1)/32] &^= 1 << ((uint32(i) - 1) & 31)
			}
		}
		updatesigmask(sigBlocked)
		for {
			select {
			case sig := <-enableSigChan:
				if b := sig - 1; b >= 0 {
					sigBlocked[b/32] &^= (1 << (b & 31))
				}
			case sig := <-disableSigChan:
				if b := sig - 1; b >= 0 {
					sigBlocked[b/32] |= (1 << (b & 31))
				}
			}
			updatesigmask(sigBlocked)
			maskUpdatedChan <- struct{}{}
		}
	}()
}
