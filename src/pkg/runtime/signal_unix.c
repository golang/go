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
		if(t->flags == 0)
			continue;
		runtime·setsig(i, runtime·sighandler, 1);
	}
}

void
runtime·resetcpuprofiler(int32 hz)
{
	Itimerval it;
	
	runtime·memclr((byte*)&it, sizeof it);
	if(hz == 0) {
		runtime·setitimer(ITIMER_PROF, &it, nil);
		runtime·setsig(SIGPROF, SIG_IGN, true);
	} else {
		runtime·setsig(SIGPROF, runtime·sighandler, true);
		it.it_interval.tv_sec = 0;
		it.it_interval.tv_usec = 1000000 / hz;
		it.it_value = it.it_interval;
		runtime·setitimer(ITIMER_PROF, &it, nil);
	}
	m->profilehz = hz;
}

void
os·sigpipe(void)
{
	runtime·setsig(SIGPIPE, SIG_DFL, false);
	runtime·raisesigpipe();
}
