// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


static struct SigTab sigtab[] = {
	/* 0 */	0, "SIGNONE: no trap",
	/* 1 */	0, "SIGHUP: terminal line hangup",
	/* 2 */	0, "SIGINT: interrupt program",
	/* 3 */	1, "SIGQUIT: quit program",
	/* 4 */	1, "SIGILL: illegal instruction",
	/* 5 */	0, "SIGTRAP: trace trap",	/* uncaught; used by panic and signal handler */
	/* 6 */	1, "SIGABRT: abort program",
	/* 7 */	1, "SIGEMT: emulate instruction executed",
	/* 8 */	1, "SIGFPE: floating-point exception",
	/* 9 */	0, "SIGKILL: kill program",
	/* 10 */	1, "SIGBUS: bus error",
	/* 11 */	1, "SIGSEGV: segmentation violation",
	/* 12 */	1, "SIGSYS: non-existent system call invoked",
	/* 13 */	0, "SIGPIPE: write on a pipe with no reader",
	/* 14 */	0, "SIGALRM: real-time timer expired",
	/* 15 */	0, "SIGTERM: software termination signal",
	/* 16 */	0, "SIGURG: urgent condition present on socket",
	/* 17 */	0, "SIGSTOP: stop",
	/* 18 */	0, "SIGTSTP: stop signal generated from keyboard",
	/* 19 */	0, "SIGCONT: continue after stop",
	/* 20 */	0, "SIGCHLD: child status has changed",
	/* 21 */	0, "SIGTTIN: background read attempted from control terminal",
	/* 22 */	0, "SIGTTOU: background write attempted to control terminal",
	/* 23 */	0, "SIGIO: I/O is possible on a descriptor",
	/* 24 */	0, "SIGXCPU: cpu time limit exceeded",
	/* 25 */	0, "SIGXFSZ: file size limit exceeded",
	/* 26 */	0, "SIGVTALRM: virtual time alarm",
	/* 27 */	0, "SIGPROF: profiling timer alarm",
	/* 28 */	0, "SIGWINCH: Window size change",
	/* 29 */	0, "SIGINFO: status request from keyboard",
	/* 30 */	0, "SIGUSR1: User defined signal 1",
	/* 31 */	0, "SIGUSR2: User defined signal 2",
};
#define	NSIG 32
