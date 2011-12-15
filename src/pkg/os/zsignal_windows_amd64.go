// MACHINE GENERATED; DO NOT EDIT
// To regenerate, run
//	./mksignals.sh
// which, for this file, will run
//	./mkunixsignals.sh ../syscall/ztypes_windows.go

package os

import (
	"syscall"
)

var _ = syscall.Open // in case there are zero signals

const (
	SIGHUP  = UnixSignal(syscall.SIGHUP)
	SIGINT  = UnixSignal(syscall.SIGINT)
	SIGQUIT = UnixSignal(syscall.SIGQUIT)
	SIGILL  = UnixSignal(syscall.SIGILL)
	SIGTRAP = UnixSignal(syscall.SIGTRAP)
	SIGABRT = UnixSignal(syscall.SIGABRT)
	SIGBUS  = UnixSignal(syscall.SIGBUS)
	SIGFPE  = UnixSignal(syscall.SIGFPE)
	SIGKILL = UnixSignal(syscall.SIGKILL)
	SIGSEGV = UnixSignal(syscall.SIGSEGV)
	SIGPIPE = UnixSignal(syscall.SIGPIPE)
	SIGALRM = UnixSignal(syscall.SIGALRM)
	SIGTERM = UnixSignal(syscall.SIGTERM)
)
