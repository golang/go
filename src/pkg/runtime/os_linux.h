// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define SS_DISABLE 2

// Linux-specific system calls
int32	runtime·futex(uint32*, int32, uint32, Timespec*, uint32*, uint32);
int32	runtime·clone(int32, void*, M*, G*, void(*)(void));

struct Sigaction;
int32	runtime·rt_sigaction(uintptr, struct Sigaction*, void*, uintptr);

void	runtime·sigaltstack(Sigaltstack*, Sigaltstack*);
void	runtime·sigpanic(void);
void runtime·setitimer(int32, Itimerval*, Itimerval*);


#define	NSIG	65
#define	SI_USER 0

// It's hard to tease out exactly how big a Sigset is, but
// rt_sigprocmask crashes if we get it wrong, so if binaries
// are running, this is right.
typedef struct Sigset Sigset;
struct Sigset
{
	uint32 mask[2];
};
void	runtime·rtsigprocmask(int32, Sigset*, Sigset*, int32);
#define SIG_SETMASK 2

#define RLIMIT_AS 9
typedef struct Rlimit Rlimit;
struct Rlimit {
	uintptr	rlim_cur;
	uintptr	rlim_max;
};
int32	runtime·getrlimit(int32, Rlimit*);
