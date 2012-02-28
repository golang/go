// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux openbsd netbsd

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"

extern SigTab runtime·sigtab[];

String
runtime·signame(int32 sig)
{
	if(sig < 0 || sig >= NSIG)
		return runtime·emptystring;
	return runtime·gostringnocopy((byte*)runtime·sigtab[sig].name);
}

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
		runtime·setsig(i, runtime·sighandler, true);
	}
}

void
runtime·sigenable(uint32 sig)
{
	int32 i;
	SigTab *t;

	for(i = 0; i<NSIG; i++) {
		// ~0 means all signals.
		if(~sig == 0 || i == sig) {
			t = &runtime·sigtab[i];
			if(t->flags & SigDefault) {
				runtime·setsig(i, runtime·sighandler, true);
				t->flags &= ~SigDefault;  // make this idempotent
			}
		}
	}
}

void
runtime·resetcpuprofiler(int32 hz)
{
	Itimerval it;

	runtime·memclr((byte*)&it, sizeof it);
	if(hz == 0) {
		runtime·setitimer(ITIMER_PROF, &it, nil);
		runtime·setprof(false);
	} else {
		it.it_interval.tv_sec = 0;
		it.it_interval.tv_usec = 1000000 / hz;
		it.it_value = it.it_interval;
		runtime·setitimer(ITIMER_PROF, &it, nil);
		runtime·setprof(true);
	}
	m->profilehz = hz;
}

void
os·sigpipe(void)
{
	runtime·setsig(SIGPIPE, SIG_DFL, false);
	runtime·raisesigpipe();
}
