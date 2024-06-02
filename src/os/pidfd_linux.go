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
	"internal/syscall/unix"
	"sync"
	"syscall"
	"unsafe"
)

func ensurePidfd(sysAttr *syscall.SysProcAttr) *syscall.SysProcAttr {
	if !pidfdWorks() {
		return sysAttr
	}

	var pidfd int

	if sysAttr == nil {
		return &syscall.SysProcAttr{
			PidFD: &pidfd,
		}
	}
	if sysAttr.PidFD == nil {
		newSys := *sysAttr // copy
		newSys.PidFD = &pidfd
		return &newSys
	}

	return sysAttr
}

func getPidfd(sysAttr *syscall.SysProcAttr) uintptr {
	if !pidfdWorks() {
		return unsetHandle
	}

	return uintptr(*sysAttr.PidFD)
}

func pidfdFind(pid int) (uintptr, error) {
	if !pidfdWorks() {
		return unsetHandle, syscall.ENOSYS
	}

	h, err := unix.PidFDOpen(pid, 0)
	if err == nil {
		return h, nil
	}
	return unsetHandle, convertESRCH(err)
}

func (p *Process) pidfdRelease() {
	// Release pidfd unconditionally.
	handle := p.handle.Swap(unsetHandle)
	if handle != unsetHandle {
		syscall.Close(int(handle))
	}
}

// _P_PIDFD is used as idtype argument to waitid syscall.
const _P_PIDFD = 3

func (p *Process) pidfdWait() (*ProcessState, error) {
	handle := p.handle.Load()
	if handle == unsetHandle || !pidfdWorks() {
		return nil, syscall.ENOSYS
	}
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
		if e == syscall.EINVAL {
			// This is either invalid option value (which should not happen
			// as we only use WEXITED), or missing P_PIDFD support (Linux
			// kernel < 5.4), meaning pidfd support is not implemented.
			e = syscall.ENOSYS
		}
		return nil, e
	}
	p.setDone()
	p.pidfdRelease()
	return &ProcessState{
		pid:    int(info.Pid),
		status: info.WaitStatus(),
		rusage: &rusage,
	}, nil
}

func (p *Process) pidfdSendSignal(s syscall.Signal) error {
	handle := p.handle.Load()
	if handle == unsetHandle || !pidfdWorks() {
		return syscall.ENOSYS
	}
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
