// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips64 mips64le
// +build linux

package runtime

type sigTabT struct {
	flags int32
	name  string
}

var sigtable = [...]sigTabT{
	/* 0 */ {0, "SIGNONE: no trap"},
	/* 1 */ {_SigNotify + _SigKill, "SIGHUP: terminal line hangup"},
	/* 2 */ {_SigNotify + _SigKill, "SIGINT: interrupt"},
	/* 3 */ {_SigNotify + _SigThrow, "SIGQUIT: quit"},
	/* 4 */ {_SigThrow + _SigUnblock, "SIGILL: illegal instruction"},
	/* 5 */ {_SigThrow + _SigUnblock, "SIGTRAP: trace trap"},
	/* 6 */ {_SigNotify + _SigThrow, "SIGABRT: abort"},
	/* 7 */ {_SigThrow, "SIGEMT"},
	/* 8 */ {_SigPanic + _SigUnblock, "SIGFPE: floating-point exception"},
	/* 9 */ {0, "SIGKILL: kill"},
	/* 10 */ {_SigPanic + _SigUnblock, "SIGBUS: bus error"},
	/* 11 */ {_SigPanic + _SigUnblock, "SIGSEGV: segmentation violation"},
	/* 12 */ {_SigNotify, "SIGSYS: bad system call"},
	/* 13 */ {_SigNotify, "SIGPIPE: write to broken pipe"},
	/* 14 */ {_SigNotify, "SIGALRM: alarm clock"},
	/* 15 */ {_SigNotify + _SigKill, "SIGTERM: termination"},
	/* 16 */ {_SigNotify, "SIGUSR1: user-defined signal 1"},
	/* 17 */ {_SigNotify, "SIGUSR2: user-defined signal 2"},
	/* 18 */ {_SigNotify + _SigUnblock, "SIGCHLD: child status has changed"},
	/* 19 */ {_SigNotify, "SIGPWR: power failure restart"},
	/* 20 */ {_SigNotify, "SIGWINCH: window size change"},
	/* 21 */ {_SigNotify, "SIGURG: urgent condition on socket"},
	/* 22 */ {_SigNotify, "SIGIO: i/o now possible"},
	/* 23 */ {0, "SIGSTOP: stop, unblockable"},
	/* 24 */ {_SigNotify + _SigDefault, "SIGTSTP: keyboard stop"},
	/* 25 */ {_SigNotify + _SigDefault, "SIGCONT: continue"},
	/* 26 */ {_SigNotify + _SigDefault, "SIGTTIN: background read from tty"},
	/* 27 */ {_SigNotify + _SigDefault, "SIGTTOU: background write to tty"},
	/* 28 */ {_SigNotify, "SIGVTALRM: virtual alarm clock"},
	/* 29 */ {_SigNotify + _SigUnblock, "SIGPROF: profiling alarm clock"},
	/* 30 */ {_SigNotify, "SIGXCPU: cpu limit exceeded"},
	/* 31 */ {_SigNotify, "SIGXFSZ: file size limit exceeded"},
	/* 32 */ {_SigSetStack + _SigUnblock, "signal 32"}, /* SIGCANCEL; see issue 6997 */
	/* 33 */ {_SigSetStack + _SigUnblock, "signal 33"}, /* SIGSETXID; see issues 3871, 9400, 12498 */
	/* 34 */ {_SigNotify, "signal 34"},
	/* 35 */ {_SigNotify, "signal 35"},
	/* 36 */ {_SigNotify, "signal 36"},
	/* 37 */ {_SigNotify, "signal 37"},
	/* 38 */ {_SigNotify, "signal 38"},
	/* 39 */ {_SigNotify, "signal 39"},
	/* 40 */ {_SigNotify, "signal 40"},
	/* 41 */ {_SigNotify, "signal 41"},
	/* 42 */ {_SigNotify, "signal 42"},
	/* 43 */ {_SigNotify, "signal 43"},
	/* 44 */ {_SigNotify, "signal 44"},
	/* 45 */ {_SigNotify, "signal 45"},
	/* 46 */ {_SigNotify, "signal 46"},
	/* 47 */ {_SigNotify, "signal 47"},
	/* 48 */ {_SigNotify, "signal 48"},
	/* 49 */ {_SigNotify, "signal 49"},
	/* 50 */ {_SigNotify, "signal 50"},
	/* 51 */ {_SigNotify, "signal 51"},
	/* 52 */ {_SigNotify, "signal 52"},
	/* 53 */ {_SigNotify, "signal 53"},
	/* 54 */ {_SigNotify, "signal 54"},
	/* 55 */ {_SigNotify, "signal 55"},
	/* 56 */ {_SigNotify, "signal 56"},
	/* 57 */ {_SigNotify, "signal 57"},
	/* 58 */ {_SigNotify, "signal 58"},
	/* 59 */ {_SigNotify, "signal 59"},
	/* 60 */ {_SigNotify, "signal 60"},
	/* 61 */ {_SigNotify, "signal 61"},
	/* 62 */ {_SigNotify, "signal 62"},
	/* 63 */ {_SigNotify, "signal 63"},
	/* 64 */ {_SigNotify, "signal 64"},
}
