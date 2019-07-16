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

import "sync"

var (
	wdmu  sync.Mutex // guards following
	wdSet bool
	wdStr string
)

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

func fixwd(paths ...string) {
	for _, path := range paths {
		if path != "" && path[0] != '/' && path[0] != '#' {
			Fixwd()
			return
		}
	}
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
	fixwd(path)
	wdmu.Lock()
	defer wdmu.Unlock()

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
