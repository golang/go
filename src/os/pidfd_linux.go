// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Support for pidfd was added during the course of a few Linux releases:
//  v5.1: pidfd_send_signal syscall;
//  v5.2: CLONE_PIDFD flag for clone syscall;
//  v5.3: pidfd_open syscall, clone3 syscall;
//  v5.4: P_PIDFD idtype support for waitid syscall;
//  v5.6: pidfd_getfd syscall.

package os

import (
	"errors"
	"internal/syscall/unix"
	"runtime"
	"sync"
	"syscall"
	"unsafe"
)

// ensurePidfd initializes the PidFD field in sysAttr if it is not already set.
// It returns the original or modified SysProcAttr struct and a flag indicating
// whether the PidFD should be duplicated before using.
func ensurePidfd(sysAttr *syscall.SysProcAttr) (*syscall.SysProcAttr, bool) {
	if !pidfdWorks() {
		return sysAttr, false
	}

	var pidfd int

	if sysAttr == nil {
		return &syscall.SysProcAttr{
			PidFD: &pidfd,
		}, false
	}
	if sysAttr.PidFD == nil {
		newSys := *sysAttr // copy
		newSys.PidFD = &pidfd
		return &newSys, false
	}

	return sysAttr, true
}

// getPidfd returns the value of sysAttr.PidFD (or its duplicate if needDup is
// set) and a flag indicating whether the value can be used.
func getPidfd(sysAttr *syscall.SysProcAttr, needDup bool) (uintptr, bool) {
	if !pidfdWorks() {
		return 0, false
	}

	h := *sysAttr.PidFD
	if needDup {
		dupH, e := unix.Fcntl(h, syscall.F_DUPFD_CLOEXEC, 0)
		if e != nil {
			return 0, false
		}
		h = dupH
	}
	return uintptr(h), true
}

func pidfdFind(pid int) (uintptr, error) {
	if !pidfdWorks() {
		return 0, syscall.ENOSYS
	}

	h, err := unix.PidFDOpen(pid, 0)
	if err != nil {
		return 0, convertESRCH(err)
	}
	return h, nil
}

// _P_PIDFD is used as idtype argument to waitid syscall.
const _P_PIDFD = 3

func (p *Process) pidfdWait() (*ProcessState, error) {
	// When pidfd is used, there is no wait/kill race (described in CL 23967)
	// because the PID recycle issue doesn't exist (IOW, pidfd, unlike PID,
	// is guaranteed to refer to one particular process). Thus, there is no
	// need for the workaround (blockUntilWaitable + sigMu) from pidWait.
	//
	// We _do_ need to be careful about reuse of the pidfd FD number when
	// closing the pidfd. See handle for more details.
	handle, status := p.handleTransientAcquire()
	switch status {
	case statusDone:
		// Process already completed Wait, or was not found by
		// pidfdFind. Return ECHILD for consistency with what the wait
		// syscall would return.
		return nil, NewSyscallError("wait", syscall.ECHILD)
	case statusReleased:
		return nil, syscall.EINVAL
	}
	defer p.handleTransientRelease()

	var (
		info   unix.SiginfoChild
		rusage syscall.Rusage
		e      syscall.Errno
	)
	for {
		_, _, e = syscall.Syscall6(syscall.SYS_WAITID, _P_PIDFD, handle, uintptr(unsafe.Pointer(&info)), syscall.WEXITED, uintptr(unsafe.Pointer(&rusage)), 0)
		if e != syscall.EINTR {
			break
		}
	}
	if e != 0 {
		return nil, NewSyscallError("waitid", e)
	}
	// Release the Process' handle reference, in addition to the reference
	// we took above.
	p.handlePersistentRelease(statusDone)
	return &ProcessState{
		pid:    int(info.Pid),
		status: info.WaitStatus(),
		rusage: &rusage,
	}, nil
}

func (p *Process) pidfdSendSignal(s syscall.Signal) error {
	handle, status := p.handleTransientAcquire()
	switch status {
	case statusDone:
		return ErrProcessDone
	case statusReleased:
		return errors.New("os: process already released")
	}
	defer p.handleTransientRelease()

	return convertESRCH(unix.PidFDSendSignal(handle, s))
}

func pidfdWorks() bool {
	return checkPidfdOnce() == nil
}

var checkPidfdOnce = sync.OnceValue(checkPidfd)

// checkPidfd checks whether all required pidfd-related syscalls work.
// This consists of pidfd_open and pidfd_send_signal syscalls, and waitid
// syscall with idtype of P_PIDFD.
//
// Reasons for non-working pidfd syscalls include an older kernel and an
// execution environment in which the above system calls are restricted by
// seccomp or a similar technology.
func checkPidfd() error {
	// In Android version < 12, pidfd_open and pidfd_send_signal are not allowed by seccomp.
	if runtime.GOOS == "android" && unix.AndroidVersion() < 12 {
		return NewSyscallError("pidfd_send_signal", syscall.ENOSYS)
	}

	// Get a pidfd of the current process (opening of "/proc/self" won't
	// work for waitid).
	fd, err := unix.PidFDOpen(syscall.Getpid(), 0)
	if err != nil {
		return NewSyscallError("pidfd_open", err)
	}
	defer syscall.Close(int(fd))

	// Check waitid(P_PIDFD) works.
	for {
		_, _, err = syscall.Syscall6(syscall.SYS_WAITID, _P_PIDFD, fd, 0, syscall.WEXITED, 0, 0)
		if err != syscall.EINTR {
			break
		}
	}
	// Expect ECHILD from waitid since we're not our own parent.
	if err != syscall.ECHILD {
		return NewSyscallError("pidfd_wait", err)
	}

	// Check pidfd_send_signal works (should be able to send 0 to itself).
	if err := unix.PidFDSendSignal(fd, 0); err != nil {
		return NewSyscallError("pidfd_send_signal", err)
	}

	return nil
}
