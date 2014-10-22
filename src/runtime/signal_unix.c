// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"
#include "signal_unix.h"

extern SigTab runtime·sigtab[];

void
runtime·initsig(void)
{
	int32 i;
	SigTab *t;

	// First call: basic setup.
	for(i = 0; i<NSIG; i++) {
		t = &runtime·sigtab[i];
		if((t->flags == 0) || (t->flags & SigDefault))
			continue;

		// For some signals, we respect an inherited SIG_IGN handler
		// rather than insist on installing our own default handler.
		// Even these signals can be fetched using the os/signal package.
		switch(i) {
		case SIGHUP:
		case SIGINT:
			if(runtime·getsig(i) == SIG_IGN) {
				t->flags = SigNotify | SigIgnored;
				continue;
			}
		}

		t->flags |= SigHandling;
		runtime·setsig(i, runtime·sighandler, true);
	}
}

void
runtime·sigenable(uint32 sig)
{
	SigTab *t;

	if(sig >= NSIG)
		return;

	t = &runtime·sigtab[sig];
	if((t->flags & SigNotify) && !(t->flags & SigHandling)) {
		t->flags |= SigHandling;
		if(runtime·getsig(sig) == SIG_IGN)
			t->flags |= SigIgnored;
		runtime·setsig(sig, runtime·sighandler, true);
	}
}

void
runtime·sigdisable(uint32 sig)
{
	SigTab *t;

	if(sig >= NSIG)
		return;

	t = &runtime·sigtab[sig];
	if((t->flags & SigNotify) && (t->flags & SigHandling)) {
		t->flags &= ~SigHandling;
		if(t->flags & SigIgnored)
			runtime·setsig(sig, SIG_IGN, true);
		else
			runtime·setsig(sig, SIG_DFL, true);
	}
}

void
runtime·resetcpuprofiler(int32 hz)
{
	Itimerval it;

	runtime·memclr((byte*)&it, sizeof it);
	if(hz == 0) {
		runtime·setitimer(ITIMER_PROF, &it, nil);
	} else {
		it.it_interval.tv_sec = 0;
		it.it_interval.tv_usec = 1000000 / hz;
		it.it_value = it.it_interval;
		runtime·setitimer(ITIMER_PROF, &it, nil);
	}
	g->m->profilehz = hz;
}

void
runtime·sigpipe(void)
{
	runtime·setsig(SIGPIPE, SIG_DFL, false);
	runtime·raise(SIGPIPE);
}

void
runtime·crash(void)
{
#ifdef GOOS_darwin
	// OS X core dumps are linear dumps of the mapped memory,
	// from the first virtual byte to the last, with zeros in the gaps.
	// Because of the way we arrange the address space on 64-bit systems,
	// this means the OS X core file will be >128 GB and even on a zippy
	// workstation can take OS X well over an hour to write (uninterruptible).
	// Save users from making that mistake.
	if(sizeof(void*) == 8)
		return;
#endif

	runtime·unblocksignals();
	runtime·setsig(SIGABRT, SIG_DFL, false);
	runtime·raise(SIGABRT);
}
