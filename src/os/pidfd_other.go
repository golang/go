// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (unix && !linux) || (js && wasm) || wasip1 || windows

package os

import "syscall"

func ensurePidfd(sysAttr *syscall.SysProcAttr) *syscall.SysProcAttr {
	return sysAttr
}

func getPidfd(_ *syscall.SysProcAttr) uintptr {
	return unsetHandle
}

func pidfdFind(_ int) (uintptr, error) {
	return unsetHandle, syscall.ENOSYS
}

func (p *Process) pidfdRelease() {}

func (_ *Process) pidfdWait() (*ProcessState, error) {
	return nil, syscall.ENOSYS
}

func (_ *Process) pidfdSendSignal(_ syscall.Signal) error {
	return syscall.ENOSYS
}
