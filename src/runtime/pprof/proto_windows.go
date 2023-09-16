// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pprof

import (
	"errors"
	"internal/syscall/windows"
	"os"
	"syscall"
)

// readMapping adds memory mapping information to the profile.
func (b *profileBuilder) readMapping() {
	snap, err := createModuleSnapshot()
	if err != nil {
		// pprof expects a map entry, so fake one, when we haven't added anything yet.
		b.addMappingEntry(0, 0, 0, "", "", true)
		return
	}
	defer func() { _ = syscall.CloseHandle(snap) }()

	var module windows.ModuleEntry32
	module.Size = uint32(windows.SizeofModuleEntry32)
	err = windows.Module32First(snap, &module)
	if err != nil {
		// pprof expects a map entry, so fake one, when we haven't added anything yet.
		b.addMappingEntry(0, 0, 0, "", "", true)
		return
	}
	for err == nil {
		exe := syscall.UTF16ToString(module.ExePath[:])
		b.addMappingEntry(
			uint64(module.ModBaseAddr),
			uint64(module.ModBaseAddr)+uint64(module.ModBaseSize),
			0,
			exe,
			peBuildID(exe),
			false,
		)
		err = windows.Module32Next(snap, &module)
	}
}

func readMainModuleMapping() (start, end uint64, exe, buildID string, err error) {
	exe, err = os.Executable()
	if err != nil {
		return 0, 0, "", "", err
	}
	snap, err := createModuleSnapshot()
	if err != nil {
		return 0, 0, "", "", err
	}
	defer func() { _ = syscall.CloseHandle(snap) }()

	var module windows.ModuleEntry32
	module.Size = uint32(windows.SizeofModuleEntry32)
	err = windows.Module32First(snap, &module)
	if err != nil {
		return 0, 0, "", "", err
	}

	return uint64(module.ModBaseAddr), uint64(module.ModBaseAddr) + uint64(module.ModBaseSize), exe, peBuildID(exe), nil
}

func createModuleSnapshot() (syscall.Handle, error) {
	for {
		snap, err := syscall.CreateToolhelp32Snapshot(windows.TH32CS_SNAPMODULE|windows.TH32CS_SNAPMODULE32, uint32(syscall.Getpid()))
		var errno syscall.Errno
		if err != nil && errors.As(err, &errno) && errno == windows.ERROR_BAD_LENGTH {
			// When CreateToolhelp32Snapshot(SNAPMODULE|SNAPMODULE32, ...) fails
			// with ERROR_BAD_LENGTH then it should be retried until it succeeds.
			continue
		}
		return snap, err
	}
}
