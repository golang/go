// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define SS_DISABLE 2

#define SIG_BLOCK 1
#define SIG_UNBLOCK 2
#define SIG_SETMASK 3

typedef uintptr kevent_udata;

struct sigaction;

void	runtime·sigpanic(void);

void	runtime·setitimer(int32, Itimerval*, Itimerval*);
void	runtime·sigaction(int32, struct Sigaction*, struct Sigaction*);
void	runtime·sigaltstack(Sigaltstack*, Sigaltstack*);
void	runtime·sigprocmask(int32, Sigset*, Sigset*);
void	runtime·unblocksignals(void);
int32	runtime·sysctl(uint32*, uint32, byte*, uintptr*, byte*, uintptr);

#define	NSIG 73 /* number of signals in runtime·SigTab array */
#define	SI_USER	0

void	runtime·raisesigpipe(void);
void	runtime·setsig(int32, void(*)(int32, Siginfo*, void*, G*), bool);
void	runtime·sighandler(int32 sig, Siginfo *info, void *context, G *gp);
void	runtime·sigpanic(void);

#define _UC_SIGMASK	0x01
#define _UC_CPU		0x04

#define RLIMIT_AS 10
typedef struct Rlimit Rlimit;
struct Rlimit {
	int64   rlim_cur;
	int64   rlim_max;
};
int32   runtime·getrlimit(int32, Rlimit*);

// Call a library function with SysV conventions,
// and switch to os stack during the call.
#pragma	varargck	countpos	runtime·sysvicall6	2
#pragma	varargck	type		runtime·sysvicall6	uintptr
#pragma	varargck	type		runtime·sysvicall6	int32
void	runtime·asmsysvicall6(void *c);
uintptr	runtime·sysvicall6(uintptr fn, int32 count, ...);

void	runtime·miniterrno(void *fn);
