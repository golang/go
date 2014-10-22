// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

#define N SigNotify
#define K SigKill
#define T SigThrow
#define P SigPanic
#define D SigDefault

#pragma dataflag NOPTR
SigTab runtime·sigtab[] = {
	/* 0 */	0, "SIGNONE: no trap",
	/* 1 */	N+K, "SIGHUP: terminal line hangup",
	/* 2 */	N+K, "SIGINT: interrupt",
	/* 3 */	N+T, "SIGQUIT: quit",
	/* 4 */	T, "SIGILL: illegal instruction",
	/* 5 */	T, "SIGTRAP: trace trap",
	/* 6 */	N+T, "SIGABRT: abort",
	/* 7 */	T, "SIGEMT: emulate instruction executed",
	/* 8 */	P, "SIGFPE: floating-point exception",
	/* 9 */	0, "SIGKILL: kill",
	/* 10 */	P, "SIGBUS: bus error",
	/* 11 */	P, "SIGSEGV: segmentation violation",
	/* 12 */	T, "SIGSYS: bad system call",
	/* 13 */	N, "SIGPIPE: write to broken pipe",
	/* 14 */	N, "SIGALRM: alarm clock",
	/* 15 */	N+K, "SIGTERM: termination",
	/* 16 */	N, "SIGURG: urgent condition on socket",
	/* 17 */	0, "SIGSTOP: stop",
	/* 18 */	N+D, "SIGTSTP: keyboard stop",
	/* 19 */	0, "SIGCONT: continue after stop",
	/* 20 */	N, "SIGCHLD: child status has changed",
	/* 21 */	N+D, "SIGTTIN: background read from tty",
	/* 22 */	N+D, "SIGTTOU: background write to tty",
	/* 23 */	N, "SIGIO: i/o now possible",
	/* 24 */	N, "SIGXCPU: cpu limit exceeded",
	/* 25 */	N, "SIGXFSZ: file size limit exceeded",
	/* 26 */	N, "SIGVTALRM: virtual alarm clock",
	/* 27 */	N, "SIGPROF: profiling alarm clock",
	/* 28 */	N, "SIGWINCH: window size change",
	/* 29 */	N, "SIGINFO: status request from keyboard",
	/* 30 */	N, "SIGUSR1: user-defined signal 1",
	/* 31 */	N, "SIGUSR2: user-defined signal 2",
};

#undef N
#undef K
#undef T
#undef P
#undef D
