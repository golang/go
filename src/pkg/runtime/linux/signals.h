// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define C SigCatch
#define I SigIgnore
#define R SigRestart

static SigTab sigtab[] = {
	/* 0 */	0, "SIGNONE: no trap",
	/* 1 */	0, "SIGHUP: terminal line hangup",
	/* 2 */	0, "SIGINT: interrupt",
	/* 3 */	C, "SIGQUIT: quit",
	/* 4 */	C, "SIGILL: illegal instruction",
	/* 5 */	C, "SIGTRAP: trace trap",
	/* 6 */	C, "SIGABRT: abort",
	/* 7 */	C, "SIGBUS: bus error",
	/* 8 */	C, "SIGFPE: floating-point exception",
	/* 9 */	0, "SIGKILL: kill",
	/* 10 */	0, "SIGUSR1: user-defined signal 1",
	/* 11 */	C, "SIGSEGV: segmentation violation",
	/* 12 */	0, "SIGUSR2: user-defined signal 2",
	/* 13 */	I, "SIGPIPE: write to broken pipe",
	/* 14 */	0, "SIGALRM: alarm clock",
	/* 15 */	0, "SIGTERM: termination",
	/* 16 */	0, "SIGSTKFLT: stack fault",
	/* 17 */	I+R, "SIGCHLD: child status has changed",
	/* 18 */	0, "SIGCONT: continue",
	/* 19 */	0, "SIGSTOP: stop, unblockable",
	/* 20 */	0, "SIGTSTP: keyboard stop",
	/* 21 */	0, "SIGTTIN: background read from tty",
	/* 22 */	0, "SIGTTOU: background write to tty",
	/* 23 */	0, "SIGURG: urgent condition on socket",
	/* 24 */	0, "SIGXCPU: cpu limit exceeded",
	/* 25 */	0, "SIGXFSZ: file size limit exceeded",
	/* 26 */	0, "SIGVTALRM: virtual alarm clock",
	/* 27 */	0, "SIGPROF: profiling alarm clock",
	/* 28 */	I+R, "SIGWINCH: window size change",
	/* 29 */	0, "SIGIO: i/o now possible",
	/* 30 */	0, "SIGPWR: power failure restart",
	/* 31 */	C, "SIGSYS: bad system call",
};
#undef C
#undef I
#undef R

#define	NSIG 32
