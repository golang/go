// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define SIG_DFL ((void*)0)
#define SIG_IGN ((void*)1)

// Linux-specific system calls
int32	runtime·futex(uint32*, int32, uint32, Timespec*, uint32*, uint32);
int32	runtime·clone(int32, void*, M*, G*, void(*)(void));

struct Sigaction;
void	runtime·rt_sigaction(uintptr, struct Sigaction*, void*, uintptr);
void	runtime·setsig(int32, void(*)(int32, Siginfo*, void*, G*), bool);
void	runtime·sighandler(int32 sig, Siginfo *info, void *context, G *gp);

void	runtime·sigaltstack(Sigaltstack*, Sigaltstack*);
void	runtime·sigpanic(void);
void runtime·setitimer(int32, Itimerval*, Itimerval*);

void	runtime·raisesigpipe(void);

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
