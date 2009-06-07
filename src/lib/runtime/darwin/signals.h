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
	/* 5 */	C, "SIGTRAP: trace trap",	/* used by panic and array out of bounds, etc. */
	/* 6 */	C, "SIGABRT: abort",
	/* 7 */	C, "SIGEMT: emulate instruction executed",
	/* 8 */	C, "SIGFPE: floating-point exception",
	/* 9 */	0, "SIGKILL: kill",
	/* 10 */	C, "SIGBUS: bus error",
	/* 11 */	C, "SIGSEGV: segmentation violation",
	/* 12 */	C, "SIGSYS: bad system call",
	/* 13 */	I, "SIGPIPE: write to broken pipe",
	/* 14 */	0, "SIGALRM: alarm clock",
	/* 15 */	0, "SIGTERM: termination",
	/* 16 */	0, "SIGURG: urgent condition on socket",
	/* 17 */	0, "SIGSTOP: stop",
	/* 18 */	0, "SIGTSTP: keyboard stop",
	/* 19 */	0, "SIGCONT: continue after stop",
	/* 20 */	I+R, "SIGCHLD: child status has changed",
	/* 21 */	0, "SIGTTIN: background read from tty",
	/* 22 */	0, "SIGTTOU: background write to tty",
	/* 23 */	0, "SIGIO: i/o now possible",
	/* 24 */	0, "SIGXCPU: cpu limit exceeded",
	/* 25 */	0, "SIGXFSZ: file size limit exceeded",
	/* 26 */	0, "SIGVTALRM: virtual alarm clock",
	/* 27 */	0, "SIGPROF: profiling alarm clock",
	/* 28 */	I+R, "SIGWINCH: window size change",
	/* 29 */	0, "SIGINFO: status request from keyboard",
	/* 30 */	0, "SIGUSR1: user-defined signal 1",
	/* 31 */	0, "SIGUSR2: user-defined signal 2",
};
#undef C
#undef I
#undef R

#define	NSIG 32
