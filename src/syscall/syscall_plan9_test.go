// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall_test

import (
	"syscall"
	"testing"
)

// testalias checks for aliasing of error strings returned by sys1 and sys2,
// which both call the function named fn in package syscall
func testalias(t *testing.T, fn string, sys1, sys2 func() error) {
	err := sys1().Error()
	errcopy := string([]byte(err))
	sys2()
	if err != errcopy {
		t.Errorf("syscall.%s error string changed from %q to %q\n", fn, errcopy, err)
	}
}

// issue 13770: errors cannot be nested in Plan 9

func TestPlan9Syserr(t *testing.T) {
	testalias(t,
		"Syscall",
		func() error {
			return syscall.Mkdir("/", 0)
		},
		func() error {
			return syscall.Mkdir("#", 0)
		})
	testalias(t,
		"Syscall6",
		func() error {
			return syscall.Mount(0, 0, "", 0, "")
		},
		func() error {
			return syscall.Mount(-1, 0, "", 0, "")
		})
	// originally failed only on plan9_arm
	testalias(t,
		"seek",
		func() error {
			_, err := syscall.Seek(0, 0, -1)
			return err
		},
		func() error {
			_, err := syscall.Seek(-1, 0, 0)
			return err
		})
}
