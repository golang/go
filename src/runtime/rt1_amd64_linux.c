// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "signals.h"

/*
 * This assembler routine takes the args from registers, puts them on the stack,
 * and calls sighandler().
 */
extern void sigtramp();

/*
 * Rudimentary reverse-engineered definition of signal interface.
 * You'd think it would be documented.
 */
typedef struct siginfo {
	int32	si_signo;		/* signal number */
	int32	si_errno;		/* errno association */
	int32	si_code;		/* signal code */
	int32	si_status;		/* exit value */
	void	*si_addr;		/* faulting address */
	/* more stuff here */
} siginfo;

typedef struct  sigaction {
 	union {
		void    (*sa_handler)(int32);
		void    (*sa_sigaction)(int32, siginfo *, void *);
	} u;		     /* signal handler */
	uint8 sa_mask[128];		     /* signal mask to apply. 128? are they MORONS? */
	int32     sa_flags;		     /* see signal options below */
	void (*sa_restorer) (void);	/* unused here; needed to return from trap? */
} sigaction;

void
sighandler(int32 sig, siginfo* info, void** context) {
	int32 i;

	if(sig < 0 || sig >= NSIG){
		prints("Signal ");
		sys·printint(sig);
	}else{
		prints(sigtab[sig].name);
	}
	prints("\nFaulting address: 0x");
	sys·printpointer(info->si_addr);
	prints("\nPC: 0x");
	sys·printpointer(context[21]);
	prints("\nSP: 0x");
	sys·printpointer(context[20]);
	prints("\n");
	traceback(context[21], context[20]);	/* empirically discovered locations */
	sys·breakpoint();
	sys·exit(2);
}

sigaction a;

void
initsig(void)
{
	int32 i;
	a.u.sa_sigaction = (void*)sigtramp;
	a.sa_flags |= 0x04;  /* SA_SIGINFO */
	for(i=0; i<sizeof(a.sa_mask); i++)
		a.sa_mask[i] = 0xFF;

	for(i = 0; i <NSIG; i++)
		if(sigtab[i].catch){
			sys·rt_sigaction(i, &a, (void*)0, 8);
		}
}
