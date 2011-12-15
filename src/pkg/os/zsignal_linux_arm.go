// MACHINE GENERATED; DO NOT EDIT
// To regenerate, run
//	./mksignals.sh
// which, for this file, will run
//	./mkunixsignals.sh ../syscall/zerrors_linux_arm.go

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
	SIGCLD    = UnixSignal(syscall.SIGCLD)
	SIGCONT   = UnixSignal(syscall.SIGCONT)
	SIGFPE    = UnixSignal(syscall.SIGFPE)
	SIGHUP    = UnixSignal(syscall.SIGHUP)
	SIGILL    = UnixSignal(syscall.SIGILL)
	SIGINT    = UnixSignal(syscall.SIGINT)
	SIGIO     = UnixSignal(syscall.SIGIO)
	SIGIOT    = UnixSignal(syscall.SIGIOT)
	SIGKILL   = UnixSignal(syscall.SIGKILL)
	SIGPIPE   = UnixSignal(syscall.SIGPIPE)
	SIGPOLL   = UnixSignal(syscall.SIGPOLL)
	SIGPROF   = UnixSignal(syscall.SIGPROF)
	SIGPWR    = UnixSignal(syscall.SIGPWR)
	SIGQUIT   = UnixSignal(syscall.SIGQUIT)
	SIGSEGV   = UnixSignal(syscall.SIGSEGV)
	SIGSTKFLT = UnixSignal(syscall.SIGSTKFLT)
	SIGSTOP   = UnixSignal(syscall.SIGSTOP)
	SIGSYS    = UnixSignal(syscall.SIGSYS)
	SIGTERM   = UnixSignal(syscall.SIGTERM)
	SIGTRAP   = UnixSignal(syscall.SIGTRAP)
	SIGTSTP   = UnixSignal(syscall.SIGTSTP)
	SIGTTIN   = UnixSignal(syscall.SIGTTIN)
	SIGTTOU   = UnixSignal(syscall.SIGTTOU)
	SIGUNUSED = UnixSignal(syscall.SIGUNUSED)
	SIGURG    = UnixSignal(syscall.SIGURG)
	SIGUSR1   = UnixSignal(syscall.SIGUSR1)
	SIGUSR2   = UnixSignal(syscall.SIGUSR2)
	SIGVTALRM = UnixSignal(syscall.SIGVTALRM)
	SIGWINCH  = UnixSignal(syscall.SIGWINCH)
	SIGXCPU   = UnixSignal(syscall.SIGXCPU)
	SIGXFSZ   = UnixSignal(syscall.SIGXFSZ)
)
