// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

typedef byte* kevent_udata;

int32	runtime·thr_new(ThrParam*, int32);
void	runtime·sigpanic(void);
void	runtime·sigaltstack(SigaltstackT*, SigaltstackT*);
struct	sigaction;
void	runtime·sigaction(int32, struct sigaction*, struct sigaction*);
void	runtime·sigprocmask(Sigset *, Sigset *);
void	runtime·unblocksignals(void);
void	runtime·setitimer(int32, Itimerval*, Itimerval*);
int32	runtime·sysctl(uint32*, uint32, byte*, uintptr*, byte*, uintptr);

enum {
	SS_DISABLE = 4,
	NSIG = 33,
	SI_USER = 0x10001,
	RLIMIT_AS = 10,
};

typedef struct Rlimit Rlimit;
struct Rlimit {
	int64	rlim_cur;
	int64	rlim_max;
};
int32	runtime·getrlimit(int32, Rlimit*);
