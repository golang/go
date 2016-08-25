// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin
// +build amd64 386 arm arm64

package syscall_test

import (
	"syscall"
	"testing"
)

func TestDarwinGettimeofday(t *testing.T) {
	tv := &syscall.Timeval{}
	if err := syscall.Gettimeofday(tv); err != nil {
		t.Fatal(err)
	}
	if tv.Sec == 0 && tv.Usec == 0 {
		t.Fatal("Sec and Usec both zero")
	}
}
