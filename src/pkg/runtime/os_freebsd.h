// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define SS_DISABLE 4

int32	runtime·thr_new(ThrParam*, int32);
void	runtime·sigpanic(void);
void	runtime·sigaltstack(Sigaltstack*, Sigaltstack*);
struct	sigaction;
void	runtime·sigaction(int32, struct sigaction*, struct sigaction*);
void	runtime·sigprocmask(Sigset *, Sigset *);
void	runtime·setitimer(int32, Itimerval*, Itimerval*);
int32	runtime·sysctl(uint32*, uint32, byte*, uintptr*, byte*, uintptr);


#define	NSIG 33
#define	SI_USER	0x10001

#define RLIMIT_AS 10
typedef struct Rlimit Rlimit;
struct Rlimit {
	int64	rlim_cur;
	int64	rlim_max;
};
int32	runtime·getrlimit(int32, Rlimit*);
