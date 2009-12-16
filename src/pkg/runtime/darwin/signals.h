// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define C SigCatch
#define I SigIgnore
#define R SigRestart
#define Q SigQueue

static SigTab sigtab[] = {
	/* 0 */	0, "SIGNONE: no trap",
	/* 1 */	Q+R, "SIGHUP: terminal line hangup",
	/* 2 */	Q+R, "SIGINT: interrupt",
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
	/* 14 */	Q+R, "SIGALRM: alarm clock",
	/* 15 */	Q+R, "SIGTERM: termination",
	/* 16 */	Q+R, "SIGURG: urgent condition on socket",
	/* 17 */	0, "SIGSTOP: stop",
	/* 18 */	Q+R, "SIGTSTP: keyboard stop",
	/* 19 */	0, "SIGCONT: continue after stop",
	/* 20 */	I+R, "SIGCHLD: child status has changed",
	/* 21 */	Q+R, "SIGTTIN: background read from tty",
	/* 22 */	Q+R, "SIGTTOU: background write to tty",
	/* 23 */	Q+R, "SIGIO: i/o now possible",
	/* 24 */	Q+R, "SIGXCPU: cpu limit exceeded",
	/* 25 */	Q+R, "SIGXFSZ: file size limit exceeded",
	/* 26 */	Q+R, "SIGVTALRM: virtual alarm clock",
	/* 27 */	Q+R, "SIGPROF: profiling alarm clock",
	/* 28 */	Q+R, "SIGWINCH: window size change",
	/* 29 */	Q+R, "SIGINFO: status request from keyboard",
	/* 30 */	Q+R, "SIGUSR1: user-defined signal 1",
	/* 31 */	Q+R, "SIGUSR2: user-defined signal 2",
};
#undef C
#undef I
#undef R
#undef Q

#define	NSIG 32
