// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import (
	"errors"
	"fmt"
	"strconv"
)

// Pledge implements the pledge syscall.
//
// This changes both the promises and execpromises; use PledgePromises or
// PledgeExecpromises to only change the promises or execpromises
// respectively.
//
// For more information see pledge(2).
func Pledge(promises, execpromises string) error {
	if err := pledgeAvailable(); err != nil {
		return err
	}

	pptr, err := BytePtrFromString(promises)
	if err != nil {
		return err
	}

	exptr, err := BytePtrFromString(execpromises)
	if err != nil {
		return err
	}

	return pledge(pptr, exptr)
}

// PledgePromises implements the pledge syscall.
//
// This changes the promises and leaves the execpromises untouched.
//
// For more information see pledge(2).
func PledgePromises(promises string) error {
	if err := pledgeAvailable(); err != nil {
		return err
	}

	pptr, err := BytePtrFromString(promises)
	if err != nil {
		return err
	}

	return pledge(pptr, nil)
}

// PledgeExecpromises implements the pledge syscall.
//
// This changes the execpromises and leaves the promises untouched.
//
// For more information see pledge(2).
func PledgeExecpromises(execpromises string) error {
	if err := pledgeAvailable(); err != nil {
		return err
	}

	exptr, err := BytePtrFromString(execpromises)
	if err != nil {
		return err
	}

	return pledge(nil, exptr)
}

// majmin returns major and minor version number for an OpenBSD system.
func majmin() (major int, minor int, err error) {
	var v Utsname
	err = Uname(&v)
	if err != nil {
		return
	}

	major, err = strconv.Atoi(string(v.Release[0]))
	if err != nil {
		err = errors.New("cannot parse major version number returned by uname")
		return
	}

	minor, err = strconv.Atoi(string(v.Release[2]))
	if err != nil {
		err = errors.New("cannot parse minor version number returned by uname")
		return
	}

	return
}

// pledgeAvailable checks for availability of the pledge(2) syscall
// based on the running OpenBSD version.
func pledgeAvailable() error {
	maj, min, err := majmin()
	if err != nil {
		return err
	}

	// Require OpenBSD 6.4 as a minimum.
	if maj < 6 || (maj == 6 && min <= 3) {
		return fmt.Errorf("cannot call Pledge on OpenBSD %d.%d", maj, min)
	}

	return nil
}
