// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


typedef uintptr kevent_udata;

struct sigaction;

void	runtime·sigpanic(void);

void	runtime·setitimer(int32, Itimerval*, Itimerval*);
void	runtime·sigaction(int32, struct SigactionT*, struct SigactionT*);
void	runtime·sigaltstack(SigaltstackT*, SigaltstackT*);
void	runtime·sigprocmask(int32, Sigset*, Sigset*);
void	runtime·unblocksignals(void);
int32	runtime·sysctl(uint32*, uint32, byte*, uintptr*, byte*, uintptr);


void	runtime·raisesigpipe(void);
void	runtime·setsig(int32, void(*)(int32, Siginfo*, void*, G*), bool);
void	runtime·sighandler(int32 sig, Siginfo *info, void *context, G *gp);
void	runtime·sigpanic(void);

enum {
	SS_DISABLE = 2,
	SIG_BLOCK = 1,
	SIG_UNBLOCK = 2,
	SIG_SETMASK = 3,
	NSIG = 73, /* number of signals in runtime·SigTab array */
	SI_USER = 0,
	_UC_SIGMASK = 0x01,
	_UC_CPU = 0x04,
	RLIMIT_AS = 10,
};

typedef struct Rlimit Rlimit;
struct Rlimit {
	int64   rlim_cur;
	int64   rlim_max;
};
int32   runtime·getrlimit(int32, Rlimit*);

// Call an external library function described by {fn, a0, ..., an}, with
// SysV conventions, switching to os stack during the call, if necessary.
uintptr	runtime·sysvicall0(uintptr fn);
uintptr	runtime·sysvicall1(uintptr fn, uintptr a1);
uintptr	runtime·sysvicall2(uintptr fn, uintptr a1, uintptr a2);
uintptr	runtime·sysvicall3(uintptr fn, uintptr a1, uintptr a2, uintptr a3);
uintptr	runtime·sysvicall4(uintptr fn, uintptr a1, uintptr a2, uintptr a3, uintptr a4);
uintptr	runtime·sysvicall5(uintptr fn, uintptr a1, uintptr a2, uintptr a3, uintptr a4, uintptr a5);
uintptr	runtime·sysvicall6(uintptr fn, uintptr a1, uintptr a2, uintptr a3, uintptr a4, uintptr a5, uintptr a6);
void	runtime·asmsysvicall6(void *c);

void	runtime·miniterrno(void *fn);
