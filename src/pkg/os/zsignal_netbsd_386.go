// MACHINE GENERATED; DO NOT EDIT
// To regenerate, run
//	./mksignals.sh
// which, for this file, will run
//	./mkunixsignals.sh ../syscall/zerrors_netbsd_386.go

package os

import (
	"syscall"
)

var _ = syscall.Open // in case there are zero signals

const (
	SIGABRT   = UnixSignal(syscall.SIGABRT)
	SIGALRM   = UnixSignal(syscall.SIGALRM)
	SIGBUS    = UnixSignal(syscall.SIGBUS)
	SIGCHLD   = UnixSignal(syscall.SIGCHLD)
	SIGCONT   = UnixSignal(syscall.SIGCONT)
	SIGEMT    = UnixSignal(syscall.SIGEMT)
	SIGFPE    = UnixSignal(syscall.SIGFPE)
	SIGHUP    = UnixSignal(syscall.SIGHUP)
	SIGILL    = UnixSignal(syscall.SIGILL)
	SIGINFO   = UnixSignal(syscall.SIGINFO)
	SIGINT    = UnixSignal(syscall.SIGINT)
	SIGIO     = UnixSignal(syscall.SIGIO)
	SIGIOT    = UnixSignal(syscall.SIGIOT)
	SIGKILL   = UnixSignal(syscall.SIGKILL)
	SIGPIPE   = UnixSignal(syscall.SIGPIPE)
	SIGPROF   = UnixSignal(syscall.SIGPROF)
	SIGQUIT   = UnixSignal(syscall.SIGQUIT)
	SIGSEGV   = UnixSignal(syscall.SIGSEGV)
	SIGSTOP   = UnixSignal(syscall.SIGSTOP)
	SIGSYS    = UnixSignal(syscall.SIGSYS)
	SIGTERM   = UnixSignal(syscall.SIGTERM)
	SIGTHR    = UnixSignal(syscall.SIGTHR)
	SIGTRAP   = UnixSignal(syscall.SIGTRAP)
	SIGTSTP   = UnixSignal(syscall.SIGTSTP)
	SIGTTIN   = UnixSignal(syscall.SIGTTIN)
	SIGTTOU   = UnixSignal(syscall.SIGTTOU)
	SIGURG    = UnixSignal(syscall.SIGURG)
	SIGUSR1   = UnixSignal(syscall.SIGUSR1)
	SIGUSR2   = UnixSignal(syscall.SIGUSR2)
	SIGVTALRM = UnixSignal(syscall.SIGVTALRM)
	SIGWINCH  = UnixSignal(syscall.SIGWINCH)
	SIGXCPU   = UnixSignal(syscall.SIGXCPU)
	SIGXFSZ   = UnixSignal(syscall.SIGXFSZ)
)
