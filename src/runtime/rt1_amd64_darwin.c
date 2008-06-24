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
	int32	si_pid;			/* sending process */
	int32	si_uid;			/* sender's ruid */
	int32	si_status;		/* exit value */
	void	*si_addr;		/* faulting address */
	/* more stuff here */
} siginfo;

typedef struct  sigaction {
 	union {
		void    (*sa_handler)(int32);
		void    (*sa_sigaction)(int32, siginfo *, void *);
	} u;		     /* signal handler */
	void	(*sa_trampoline)(void);	/* kernel callback point; calls sighandler() */
	uint8 sa_mask[4];		     /* signal mask to apply */
	int32     sa_flags;		     /* see signal options below */
} sigaction;

void
sighandler(int32 sig, siginfo* info, void** context) {
	int32 i;
	void *pc, *sp;

	if(sig < 0 || sig >= NSIG){
		prints("Signal ");
		sys_printint(sig);
	}else{
		prints(sigtab[sig].name);
	}

	prints("\nFaulting address: 0x");
	sys_printpointer(info->si_addr);
	prints("\nPC: 0x");
	pc = ((void**)((&sig)+1))[22];
	sys_printpointer(pc);
	prints("\nSP: 0x");
	sp = ((void**)((&sig)+1))[13];
	sys_printpointer(sp);
	prints("\n");
	if (pc != 0 && sp != 0)
		traceback(pc, sp);	/* empirically discovered locations */
	sys_exit(2);
}

sigaction a;
extern void sigtramp(void);

void
initsig(void)
{
	int32 i;
	a.u.sa_sigaction = (void*)sigtramp;
	a.sa_flags |= 0x40;  /* SA_SIGINFO */
	for(i=0; i<sizeof(a.sa_mask); i++)
		a.sa_mask[i] = 0xFF;
	a.sa_trampoline = sigtramp;

	for(i = 0; i <NSIG; i++)
		if(sigtab[i].catch){
			sys_sigaction(i, &a, (void*)0);
		}
}
