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
	int32     sa_flags;		     /* see signal options below */
	uint8 sa_mask[2];		     /* signal mask to apply. BUG: 2 is a guess */
} sigaction;

void
sighandler(int32 sig, siginfo* info, void** context) {
	int32 i;

	if(sig < 0 || sig >= NSIG){
		prints("Signal ");
		sys_printint(sig);
	}else{
		prints(sigtab[sig].name);
	}
	prints("\nFaulting address: 0x");
	sys_printpointer(info->si_addr);
	prints("\nPC: 0x");
	sys_printpointer(context[21]);
	prints("\nSP: 0x");
	sys_printpointer(context[20]);
	prints("\n");
	traceback(context[21], context[20]);	/* empirically discovered locations */
	sys_breakpoint();
	sys_exit(2);
}

sigaction a;

void
initsig(void)
{
	int32 i;
	a.u.sa_sigaction = (void*)sigtramp;
	a.sa_flags = 1|2|4|0x10000000|0x20000000|0x40000000|0x80000000;
	//a.sa_flags |= SA_SIGINFO;
	a.sa_flags = ~0;	/* BUG: why is this needed? */
	for(i=0; i<sizeof(a.sa_mask); i++)
		a.sa_mask[i] = 0xFF;
	//a.sa_mask[1] = (1 << (11-1));
	for(i = 0; i <NSIG; i++)
		if(sigtab[i].catch){
			sys_rt_sigaction(i, &a, (void*)0, 8);
		}
}
