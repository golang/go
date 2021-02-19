// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows
// +build windows

package main

import (
	"context"
	"log"
	"os"
	"syscall"
	"unsafe"
)

const (
	envSeparator = ";"
	homeKey      = "USERPROFILE"
	lineEnding   = "/r/n"
	pathVar      = "$env:Path"
)

var installPath = `c:\go`

func isWindowsXP() bool {
	v, err := syscall.GetVersion()
	if err != nil {
		log.Fatalf("GetVersion failed: %v", err)
	}
	major := byte(v)
	return major < 6
}

func whichGo(ctx context.Context) (string, error) {
	return findGo(ctx, "where")
}

// currentShell reports the current shell.
// It might be "powershell.exe", "cmd.exe" or any of the *nix shells.
//
// Returns empty string if the shell is unknown.
func currentShell() string {
	shell := os.Getenv("SHELL")
	if shell != "" {
		return shell
	}

	pid := os.Getppid()
	pe, err := getProcessEntry(pid)
	if err != nil {
		verbosef("getting shell from process entry failed: %v", err)
		return ""
	}

	return syscall.UTF16ToString(pe.ExeFile[:])
}

func getProcessEntry(pid int) (*syscall.ProcessEntry32, error) {
	// From https://go.googlesource.com/go/+/go1.8.3/src/syscall/syscall_windows.go#941
	snapshot, err := syscall.CreateToolhelp32Snapshot(syscall.TH32CS_SNAPPROCESS, 0)
	if err != nil {
		return nil, err
	}
	defer syscall.CloseHandle(snapshot)

	var procEntry syscall.ProcessEntry32
	procEntry.Size = uint32(unsafe.Sizeof(procEntry))
	if err = syscall.Process32First(snapshot, &procEntry); err != nil {
		return nil, err
	}

	for {
		if procEntry.ProcessID == uint32(pid) {
			return &procEntry, nil
		}

		if err := syscall.Process32Next(snapshot, &procEntry); err != nil {
			return nil, err
		}
	}
}

func persistEnvChangesForSession() error {
	return nil
}
