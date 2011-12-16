// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define SIG_DFL ((void*)0)
#define SIG_IGN ((void*)1)

struct sigaction;

void	runtime·sigpanic(void);
void	runtime·sigaltstack(Sigaltstack*, Sigaltstack*);
void	runtime·sigaction(int32, struct sigaction*, struct sigaction*);
void	runtime·setitimerval(int32, Itimerval*, Itimerval*);
void	runtime·setitimer(int32, Itimerval*, Itimerval*);
int32	runtime·sysctl(uint32*, uint32, byte*, uintptr*, byte*, uintptr);

void	runtime·raisesigpipe(void);
