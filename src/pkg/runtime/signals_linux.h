// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define N SigNotify
#define K SigKill
#define T SigThrow
#define P SigPanic
#define D SigDefault

SigTab runtime·sigtab[] = {
	/* 0 */	0, "SIGNONE: no trap",
	/* 1 */	N+K, "SIGHUP: terminal line hangup",
	/* 2 */	N+K, "SIGINT: interrupt",
	/* 3 */	N+T, "SIGQUIT: quit",
	/* 4 */	T, "SIGILL: illegal instruction",
	/* 5 */	T, "SIGTRAP: trace trap",
	/* 6 */	N+T, "SIGABRT: abort",
	/* 7 */	P, "SIGBUS: bus error",
	/* 8 */	P, "SIGFPE: floating-point exception",
	/* 9 */	0, "SIGKILL: kill",
	/* 10 */	N, "SIGUSR1: user-defined signal 1",
	/* 11 */	P, "SIGSEGV: segmentation violation",
	/* 12 */	N, "SIGUSR2: user-defined signal 2",
	/* 13 */	N, "SIGPIPE: write to broken pipe",
	/* 14 */	N, "SIGALRM: alarm clock",
	/* 15 */	N+K, "SIGTERM: termination",
	/* 16 */	T, "SIGSTKFLT: stack fault",
	/* 17 */	N, "SIGCHLD: child status has changed",
	/* 18 */	0, "SIGCONT: continue",
	/* 19 */	0, "SIGSTOP: stop, unblockable",
	/* 20 */	N+D, "SIGTSTP: keyboard stop",
	/* 21 */	N+D, "SIGTTIN: background read from tty",
	/* 22 */	N+D, "SIGTTOU: background write to tty",
	/* 23 */	N, "SIGURG: urgent condition on socket",
	/* 24 */	N, "SIGXCPU: cpu limit exceeded",
	/* 25 */	N, "SIGXFSZ: file size limit exceeded",
	/* 26 */	N, "SIGVTALRM: virtual alarm clock",
	/* 27 */	N, "SIGPROF: profiling alarm clock",
	/* 28 */	N, "SIGWINCH: window size change",
	/* 29 */	N, "SIGIO: i/o now possible",
	/* 30 */	N, "SIGPWR: power failure restart",
	/* 31 */	N, "SIGSYS: bad system call",
	/* 32 */	N, "signal 32",
	/* 33 */	0, "signal 33", /* SIGSETXID; see issue 3871 */
	/* 34 */	N, "signal 34",
	/* 35 */	N, "signal 35",
	/* 36 */	N, "signal 36",
	/* 37 */	N, "signal 37",
	/* 38 */	N, "signal 38",
	/* 39 */	N, "signal 39",
	/* 40 */	N, "signal 40",
	/* 41 */	N, "signal 41",
	/* 42 */	N, "signal 42",
	/* 43 */	N, "signal 43",
	/* 44 */	N, "signal 44",
	/* 45 */	N, "signal 45",
	/* 46 */	N, "signal 46",
	/* 47 */	N, "signal 47",
	/* 48 */	N, "signal 48",
	/* 49 */	N, "signal 49",
	/* 50 */	N, "signal 50",
	/* 51 */	N, "signal 51",
	/* 52 */	N, "signal 52",
	/* 53 */	N, "signal 53",
	/* 54 */	N, "signal 54",
	/* 55 */	N, "signal 55",
	/* 56 */	N, "signal 56",
	/* 57 */	N, "signal 57",
	/* 58 */	N, "signal 58",
	/* 59 */	N, "signal 59",
	/* 60 */	N, "signal 60",
	/* 61 */	N, "signal 61",
	/* 62 */	N, "signal 62",
	/* 63 */	N, "signal 63",
	/* 64 */	N, "signal 64",
};

#undef N
#undef K
#undef T
#undef P
#undef D
