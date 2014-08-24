// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements runtime support for signal handling.
//
// Most synchronization primitives are not available from
// the signal handler (it cannot block, allocate memory, or use locks)
// so the handler communicates with a processing goroutine
// via struct sig, below.
//
// sigsend() is called by the signal handler to queue a new signal.
// signal_recv() is called by the Go program to receive a newly queued signal.
// Synchronization between sigsend() and signal_recv() is based on the sig.state
// variable.  It can be in 3 states: 0, HASWAITER and HASSIGNAL.
// HASWAITER means that signal_recv() is blocked on sig.Note and there are no
// new pending signals.
// HASSIGNAL means that sig.mask *may* contain new pending signals,
// signal_recv() can't be blocked in this state.
// 0 means that there are no new pending signals and signal_recv() is not blocked.
// Transitions between states are done atomically with CAS.
// When signal_recv() is unblocked, it resets sig.Note and rechecks sig.mask.
// If several sigsend()'s and signal_recv() execute concurrently, it can lead to
// unnecessary rechecks of sig.mask, but must not lead to missed signals
// nor deadlocks.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"
#include "cgocall.h"
#include "../../cmd/ld/textflag.h"

typedef struct Sig Sig;
struct Sig {
	uint32 mask[(NSIG+31)/32];
	uint32 wanted[(NSIG+31)/32];
	uint32 recv[(NSIG+31)/32];
	uint32 state;
	bool inuse;
	bool afterwait;
};

#pragma dataflag NOPTR
static Sig sig;

Note runtime·signote;

enum {
	HASWAITER = 1,
	HASSIGNAL = 2,
};

// Called from sighandler to send a signal back out of the signal handling thread.
bool
runtime·sigsend(int32 s)
{
	uint32 bit, mask, old, new;

	if(!sig.inuse || s < 0 || s >= 32*nelem(sig.wanted) || !(sig.wanted[s/32]&(1U<<(s&31))))
		return false;
	bit = 1 << (s&31);
	for(;;) {
		mask = sig.mask[s/32];
		if(mask & bit)
			break;		// signal already in queue
		if(runtime·cas(&sig.mask[s/32], mask, mask|bit)) {
			// Added to queue.
			// Only send a wakeup if the receiver needs a kick.
			for(;;) {
				old = runtime·atomicload(&sig.state);
				if(old == HASSIGNAL)
					break;
				if(old == HASWAITER)
					new = 0;
				else  // if(old == 0)
					new = HASSIGNAL;
				if(runtime·cas(&sig.state, old, new)) {
					if (old == HASWAITER)
						runtime·notewakeup(&runtime·signote);
					break;
				}
			}
			break;
		}
	}
	return true;
}

// Called to receive the next queued signal.
// Must only be called from a single goroutine at a time.
void
runtime·signal_recv_m(void)
{
	uint32 i, old, new;

	if(sig.afterwait) {
		sig.afterwait = false;
		goto update;
	}
	for(;;) {
		// Serve from local copy if there are bits left.
		for(i=0; i<NSIG; i++) {
			if(sig.recv[i/32]&(1U<<(i&31))) {
				sig.recv[i/32] ^= 1U<<(i&31);
				g->m->scalararg[0] = true;
				g->m->scalararg[1] = i;
				return;
			}
		}

		// Check and update sig.state.
		for(;;) {
			old = runtime·atomicload(&sig.state);
			if(old == HASWAITER)
				runtime·throw("inconsistent state in signal_recv");
			if(old == HASSIGNAL)
				new = 0;
			else  // if(old == 0)
				new = HASWAITER;
			if(runtime·cas(&sig.state, old, new)) {
				if (new == HASWAITER) {
					sig.afterwait = true;
					g->m->scalararg[0] = false;
					g->m->scalararg[1] = 0;
					return;
				}
				break;
			}
		}

		// Get a new local copy.
	update:
		for(i=0; i<nelem(sig.mask); i++) {
			for(;;) {
				old = sig.mask[i];
				if(runtime·cas(&sig.mask[i], old, 0))
					break;
			}
			sig.recv[i] = old;
		}
	}
}

// Must only be called from a single goroutine at a time.
void
runtime·signal_enable_m(void)
{
	uint32 s;

	if(!sig.inuse) {
		// The first call to signal_enable is for us
		// to use for initialization.  It does not pass
		// signal information in m.
		sig.inuse = true;	// enable reception of signals; cannot disable
		runtime·noteclear(&runtime·signote);
		return;
	}
	s = g->m->scalararg[0];
	if(s >= nelem(sig.wanted)*32)
		return;
	sig.wanted[s/32] |= 1U<<(s&31);
	runtime·sigenable(s);
}

// Must only be called from a single goroutine at a time.
void
runtime·signal_disable_m(void)
{
	uint32 s;

	s = g->m->scalararg[0];
	if(s >= nelem(sig.wanted)*32)
		return;
	sig.wanted[s/32] &= ~(1U<<(s&31));
	runtime·sigdisable(s);
}

// This runs on a foreign stack, without an m or a g.  No stack split.
#pragma textflag NOSPLIT
void
runtime·badsignal(uintptr sig)
{
	runtime·cgocallback((void (*)(void))runtime·sigsend, &sig, sizeof(sig));
}
