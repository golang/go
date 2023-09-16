// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The working directory in Plan 9 is effectively per P, so different
// goroutines and even the same goroutine as it's rescheduled on
// different Ps can see different working directories.
//
// Instead, track a Go process-wide intent of the current working directory,
// and switch to it at important points.

package syscall

import (
	"runtime"
	"sync"
)

var (
	wdmu  sync.Mutex // guards following
	wdSet bool
	wdStr string
)

// Ensure current working directory seen by this goroutine matches
// the most recent Chdir called in any goroutine. It's called internally
// before executing any syscall which uses a relative pathname. Must
// be called with the goroutine locked to the OS thread, to prevent
// rescheduling on a different thread (potentially with a different
// working directory) before the syscall is executed.
func Fixwd() {
	wdmu.Lock()
	defer wdmu.Unlock()
	fixwdLocked()
}

func fixwdLocked() {
	if !wdSet {
		return
	}
	// always call chdir when getwd returns an error
	wd, _ := getwd()
	if wd == wdStr {
		return
	}
	if err := chdir(wdStr); err != nil {
		return
	}
}

// If any of the paths is relative, call Fixwd and return true
// (locked to OS thread). Otherwise return false.
func fixwd(paths ...string) bool {
	for _, path := range paths {
		if path != "" && path[0] != '/' && path[0] != '#' {
			runtime.LockOSThread()
			Fixwd()
			return true
		}
	}
	return false
}

// goroutine-specific getwd
func getwd() (wd string, err error) {
	fd, err := open(".", O_RDONLY)
	if err != nil {
		return "", err
	}
	defer Close(fd)
	return Fd2path(fd)
}

func Getwd() (wd string, err error) {
	wdmu.Lock()
	defer wdmu.Unlock()

	if wdSet {
		return wdStr, nil
	}
	wd, err = getwd()
	if err != nil {
		return
	}
	wdSet = true
	wdStr = wd
	return wd, nil
}

func Chdir(path string) error {
	// If Chdir is to a relative path, sync working dir first
	if fixwd(path) {
		defer runtime.UnlockOSThread()
	}
	wdmu.Lock()
	defer wdmu.Unlock()

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := chdir(path); err != nil {
		return err
	}

	wd, err := getwd()
	if err != nil {
		return err
	}
	wdSet = true
	wdStr = wd
	return nil
}
