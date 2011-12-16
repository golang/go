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

void	runtime·sigaltstack(Sigaltstack*, Sigaltstack*);
void	runtime·sigpanic(void);
void runtime·setitimer(int32, Itimerval*, Itimerval*);

void	runtime·raisesigpipe(void);
