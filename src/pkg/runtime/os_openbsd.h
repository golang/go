// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define SS_DISABLE 4

#define SIG_BLOCK 1
#define SIG_UNBLOCK 2
#define SIG_SETMASK 3

typedef byte* kevent_udata;

struct sigaction;

void	runtime·sigpanic(void);

void	runtime·setitimer(int32, Itimerval*, Itimerval*);
void	runtime·sigaction(int32, struct sigaction*, struct sigaction*);
void	runtime·sigaltstack(SigaltstackT*, SigaltstackT*);
Sigset	runtime·sigprocmask(int32, Sigset);
void	runtime·unblocksignals(void);
int32	runtime·sysctl(uint32*, uint32, byte*, uintptr*, byte*, uintptr);

#define	NSIG 33
#define	SI_USER	0
