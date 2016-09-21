// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package runtime

import (
	"runtime/internal/sys"
	"unsafe"
)

//go:linkname os_sigpipe os.sigpipe
func os_sigpipe() {
	systemstack(sigpipe)
}

func signame(sig uint32) string {
	if sig >= uint32(len(sigtable)) {
		return ""
	}
	return sigtable[sig].name
}

const (
	_SIG_DFL uintptr = 0
	_SIG_IGN uintptr = 1
)

// Stores the signal handlers registered before Go installed its own.
// These signal handlers will be invoked in cases where Go doesn't want to
// handle a particular signal (e.g., signal occurred on a non-Go thread).
// See sigfwdgo() for more information on when the signals are forwarded.
//
// Signal forwarding is currently available only on Darwin and Linux.
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

func init() {
	// _NSIG is the number of signals on this operating system.
	// sigtable should describe what to do for all the possible signals.
	if len(sigtable) != _NSIG {
		print("runtime: len(sigtable)=", len(sigtable), " _NSIG=", _NSIG, "\n")
		throw("bad sigtable len")
	}
}

var signalsOK bool

// Initialize signals.
// Called by libpreinit so runtime may not be initialized.
//go:nosplit
//go:nowritebarrierrec
func initsig(preinit bool) {
	if !preinit {
		// It's now OK for signal handlers to run.
		signalsOK = true
	}

	// For c-archive/c-shared this is called by libpreinit with
	// preinit == true.
	if (isarchive || islibrary) && !preinit {
		return
	}

	for i := int32(0); i < _NSIG; i++ {
		t := &sigtable[i]
		if t.flags == 0 || t.flags&_SigDefault != 0 {
			continue
		}
		fwdSig[i] = getsig(i)

		if !sigInstallGoHandler(i) {
			// Even if we are not installing a signal handler,
			// set SA_ONSTACK if necessary.
			if fwdSig[i] != _SIG_DFL && fwdSig[i] != _SIG_IGN {
				setsigstack(i)
			}
			continue
		}

		t.flags |= _SigHandling
		setsig(i, funcPC(sighandler), true)
	}
}

//go:nosplit
//go:nowritebarrierrec
func sigInstallGoHandler(sig int32) bool {
	// For some signals, we respect an inherited SIG_IGN handler
	// rather than insist on installing our own default handler.
	// Even these signals can be fetched using the os/signal package.
	switch sig {
	case _SIGHUP, _SIGINT:
		if fwdSig[sig] == _SIG_IGN {
			return false
		}
	}

	t := &sigtable[sig]
	if t.flags&_SigSetStack != 0 {
		return false
	}

	// When built using c-archive or c-shared, only install signal
	// handlers for synchronous signals.
	if (isarchive || islibrary) && t.flags&_SigPanic == 0 {
		return false
	}

	return true
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
			fwdSig[sig] = getsig(int32(sig))
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

		// If initsig does not install a signal handler for a
		// signal, then to go back to the state before Notify
		// we should remove the one we installed.
		if !sigInstallGoHandler(int32(sig)) {
			t.flags &^= _SigHandling
			setsig(int32(sig), fwdSig[sig], true)
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
	if sigsend(_SIGPIPE) {
		return
	}
	dieFromSignal(_SIGPIPE)
}

// dieFromSignal kills the program with a signal.
// This provides the expected exit status for the shell.
// This is only called with fatal signals expected to kill the process.
//go:nosplit
//go:nowritebarrierrec
func dieFromSignal(sig int32) {
	setsig(sig, _SIG_DFL, false)
	updatesigmask(sigmask{})
	raise(sig)

	// That should have killed us. On some systems, though, raise
	// sends the signal to the whole process rather than to just
	// the current thread, which means that the signal may not yet
	// have been delivered. Give other threads a chance to run and
	// pick up the signal.
	osyield()
	osyield()
	osyield()

	// If we are still somehow running, just exit with the wrong status.
	exit(2)
}

// raisebadsignal is called when a signal is received on a non-Go
// thread, and the Go program does not want to handle it (that is, the
// program has not called os/signal.Notify for the signal).
func raisebadsignal(sig int32, c *sigctxt) {
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
	// signal is blocked. We need to unblock it before raising the
	// signal, or the signal we raise will be ignored until we return
	// from the signal handler. We know that the signal was unblocked
	// before entering the handler, or else we would not have received
	// it. That means that we don't have to worry about blocking it
	// again.
	unblocksig(sig)
	setsig(sig, handler, false)

	// If we're linked into a non-Go program we want to try to
	// avoid modifying the original context in which the signal
	// was raised. If the handler is the default, we know it
	// is non-recoverable, so we don't have to worry about
	// re-installing sighandler. At this point we can just
	// return and the signal will be re-raised and caught by
	// the default handler with the correct context.
	if (isarchive || islibrary) && handler == _SIG_DFL && c.sigcode() != _SI_USER {
		return
	}

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
		if sys.PtrSize == 8 {
			return
		}
	}

	dieFromSignal(_SIGABRT)
}

// ensureSigM starts one global, sleeping thread to make sure at least one thread
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
				if b := sig - 1; sig > 0 {
					sigBlocked[b/32] &^= (1 << (b & 31))
				}
			case sig := <-disableSigChan:
				if b := sig - 1; sig > 0 {
					sigBlocked[b/32] |= (1 << (b & 31))
				}
			}
			updatesigmask(sigBlocked)
			maskUpdatedChan <- struct{}{}
		}
	}()
}

// This is called when we receive a signal when there is no signal stack.
// This can only happen if non-Go code calls sigaltstack to disable the
// signal stack. This is called via cgocallback to establish a stack.
func noSignalStack(sig uint32) {
	println("signal", sig, "received on thread with no signal stack")
	throw("non-Go code disabled sigaltstack")
}

// This is called if we receive a signal when there is a signal stack
// but we are not on it. This can only happen if non-Go code called
// sigaction without setting the SS_ONSTACK flag.
func sigNotOnStack(sig uint32) {
	println("signal", sig, "received but handler not on signal stack")
	throw("non-Go code set up signal handler without SA_ONSTACK flag")
}

// This runs on a foreign stack, without an m or a g. No stack split.
//go:nosplit
//go:norace
//go:nowritebarrierrec
func badsignal(sig uintptr, c *sigctxt) {
	cgocallback(unsafe.Pointer(funcPC(badsignalgo)), noescape(unsafe.Pointer(&sig)), unsafe.Sizeof(sig)+unsafe.Sizeof(c), 0)
}

func badsignalgo(sig uintptr, c *sigctxt) {
	if !sigsend(uint32(sig)) {
		// A foreign thread received the signal sig, and the
		// Go code does not want to handle it.
		raisebadsignal(int32(sig), c)
	}
}

//go:noescape
func sigfwd(fn uintptr, sig uint32, info *siginfo, ctx unsafe.Pointer)

// Determines if the signal should be handled by Go and if not, forwards the
// signal to the handler that was installed before Go's. Returns whether the
// signal was forwarded.
// This is called by the signal handler, and the world may be stopped.
//go:nosplit
//go:nowritebarrierrec
func sigfwdgo(sig uint32, info *siginfo, ctx unsafe.Pointer) bool {
	if sig >= uint32(len(sigtable)) {
		return false
	}
	fwdFn := fwdSig[sig]

	if !signalsOK {
		// The only way we can get here is if we are in a
		// library or archive, we installed a signal handler
		// at program startup, but the Go runtime has not yet
		// been initialized.
		if fwdFn == _SIG_DFL {
			dieFromSignal(int32(sig))
		} else {
			sigfwd(fwdFn, sig, info, ctx)
		}
		return true
	}

	flags := sigtable[sig].flags

	// If there is no handler to forward to, no need to forward.
	if fwdFn == _SIG_DFL {
		return false
	}

	// If we aren't handling the signal, forward it.
	if flags&_SigHandling == 0 {
		sigfwd(fwdFn, sig, info, ctx)
		return true
	}

	// Only forward synchronous signals.
	c := &sigctxt{info, ctx}
	if c.sigcode() == _SI_USER || flags&_SigPanic == 0 {
		return false
	}
	// Determine if the signal occurred inside Go code. We test that:
	//   (1) we were in a goroutine (i.e., m.curg != nil), and
	//   (2) we weren't in CGO (i.e., m.curg.syscallsp == 0).
	g := getg()
	if g != nil && g.m != nil && g.m.curg != nil && g.m.curg.syscallsp == 0 {
		return false
	}
	// Signal not handled by Go, forward it.
	if fwdFn != _SIG_IGN {
		sigfwd(fwdFn, sig, info, ctx)
	}
	return true
}
