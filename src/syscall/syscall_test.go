// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall_test

import (
	"fmt"
	"syscall"
	"testing"
)

func testSetGetenv(t *testing.T, key, value string) {
	err := syscall.Setenv(key, value)
	if err != nil {
		t.Fatalf("Setenv failed to set %q: %v", value, err)
	}
	newvalue, found := syscall.Getenv(key)
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
	s := syscall.Itoa(i)
	f := fmt.Sprint(i)
	if s != f {
		t.Fatalf("itoa(%d) = %s, want %s", i, s, f)
	}
}
