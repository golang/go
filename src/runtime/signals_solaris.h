// Copyright 2014 The Go Authors. All rights reserved.
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
	/* 0 */		0, "SIGNONE: no trap",
	/* 1 */		N+K, "SIGHUP: hangup",
	/* 2 */		N+K, "SIGINT: interrupt (rubout)",
	/* 3 */		N+T, "SIGQUIT: quit (ASCII FS)",
	/* 4 */		T, "SIGILL: illegal instruction (not reset when caught)",
	/* 5 */		T, "SIGTRAP: trace trap (not reset when caught)",
	/* 6 */		N+T, "SIGABRT: used by abort, replace SIGIOT in the future",
	/* 7 */		T, "SIGEMT: EMT instruction",
	/* 8 */		P, "SIGFPE: floating point exception",
	/* 9 */		0, "SIGKILL: kill (cannot be caught or ignored)",
	/* 10 */	P, "SIGBUS: bus error",
	/* 11 */	P, "SIGSEGV: segmentation violation",
	/* 12 */	T, "SIGSYS: bad argument to system call",
	/* 13 */	N, "SIGPIPE: write on a pipe with no one to read it",
	/* 14 */	N, "SIGALRM: alarm clock",
	/* 15 */	N+K, "SIGTERM: software termination signal from kill",
	/* 16 */	N, "SIGUSR1: user defined signal 1",
	/* 17 */	N, "SIGUSR2: user defined signal 2",
	/* 18 */	N, "SIGCLD: child status change",
	/* 18 */	N, "SIGCHLD: child status change alias (POSIX)",
	/* 19 */	N, "SIGPWR: power-fail restart",
	/* 20 */	N, "SIGWINCH: window size change",
	/* 21 */	N, "SIGURG: urgent socket condition",
	/* 22 */	N, "SIGPOLL: pollable event occured",
	/* 23 */	N+D, "SIGSTOP: stop (cannot be caught or ignored)",
	/* 24 */	0, "SIGTSTP: user stop requested from tty",
	/* 25 */	0, "SIGCONT: stopped process has been continued",
	/* 26 */	N+D, "SIGTTIN: background tty read attempted",
	/* 27 */	N+D, "SIGTTOU: background tty write attempted",
	/* 28 */	N, "SIGVTALRM: virtual timer expired",
	/* 29 */	N, "SIGPROF: profiling timer expired",
	/* 30 */	N, "SIGXCPU: exceeded cpu limit",
	/* 31 */	N, "SIGXFSZ: exceeded file size limit",
	/* 32 */	N, "SIGWAITING: reserved signal no longer used by",
	/* 33 */	N, "SIGLWP: reserved signal no longer used by",
	/* 34 */	N, "SIGFREEZE: special signal used by CPR",
	/* 35 */	N, "SIGTHAW: special signal used by CPR",
	/* 36 */	0, "SIGCANCEL: reserved signal for thread cancellation",
	/* 37 */	N, "SIGLOST: resource lost (eg, record-lock lost)",
	/* 38 */	N, "SIGXRES: resource control exceeded",
	/* 39 */	N, "SIGJVM1: reserved signal for Java Virtual Machine",
	/* 40 */	N, "SIGJVM2: reserved signal for Java Virtual Machine",

	/* TODO(aram): what should be do about these signals? D or N? is this set static? */
	/* 41 */	N, "real time signal",
	/* 42 */	N, "real time signal",
	/* 43 */	N, "real time signal",
	/* 44 */	N, "real time signal",
	/* 45 */	N, "real time signal",
	/* 46 */	N, "real time signal",
	/* 47 */	N, "real time signal",
	/* 48 */	N, "real time signal",
	/* 49 */	N, "real time signal",
	/* 50 */	N, "real time signal",
	/* 51 */	N, "real time signal",
	/* 52 */	N, "real time signal",
	/* 53 */	N, "real time signal",
	/* 54 */	N, "real time signal",
	/* 55 */	N, "real time signal",
	/* 56 */	N, "real time signal",
	/* 57 */	N, "real time signal",
	/* 58 */	N, "real time signal",
	/* 59 */	N, "real time signal",
	/* 60 */	N, "real time signal",
	/* 61 */	N, "real time signal",
	/* 62 */	N, "real time signal",
	/* 63 */	N, "real time signal",
	/* 64 */	N, "real time signal",
	/* 65 */	N, "real time signal",
	/* 66 */	N, "real time signal",
	/* 67 */	N, "real time signal",
	/* 68 */	N, "real time signal",
	/* 69 */	N, "real time signal",
	/* 70 */	N, "real time signal",
	/* 71 */	N, "real time signal",
	/* 72 */	N, "real time signal",
};

#undef N
#undef K
#undef T
#undef P
#undef D
