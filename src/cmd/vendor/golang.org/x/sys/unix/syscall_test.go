// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris

package unix_test

import (
	"fmt"
	"testing"

	"golang.org/x/sys/unix"
)

func testSetGetenv(t *testing.T, key, value string) {
	err := unix.Setenv(key, value)
	if err != nil {
		t.Fatalf("Setenv failed to set %q: %v", value, err)
	}
	newvalue, found := unix.Getenv(key)
	if !found {
		t.Fatalf("Getenv failed to find %v variable (want value %q)", key, value)
	}
	if newvalue != value {
		t.Fatalf("Getenv(%v) = %q; want %q", key, newvalue, value)
	}
}

func TestEnv(t *testing.T) {
	testSetGetenv(t, "TESTENV", "AVALUE")
	// make sure TESTENV gets set to "", not deleted
	testSetGetenv(t, "TESTENV", "")
}

func TestItoa(t *testing.T) {
	// Make most negative integer: 0x8000...
	i := 1
	for i<<1 != 0 {
		i <<= 1
	}
	if i >= 0 {
		t.Fatal("bad math")
	}
	s := unix.Itoa(i)
	f := fmt.Sprint(i)
	if s != f {
		t.Fatalf("itoa(%d) = %s, want %s", i, s, f)
	}
}

func TestUname(t *testing.T) {
	var utsname unix.Utsname
	err := unix.Uname(&utsname)
	if err != nil {
		t.Fatalf("Uname: %v", err)
	}

	t.Logf("OS: %s/%s %s", utsname.Sysname[:], utsname.Machine[:], utsname.Release[:])
}
