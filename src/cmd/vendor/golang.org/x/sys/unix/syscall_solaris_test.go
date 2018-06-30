// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build solaris

package unix_test

import (
	"os/exec"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

func TestSelect(t *testing.T) {
	err := unix.Select(0, nil, nil, nil, &unix.Timeval{Sec: 0, Usec: 0})
	if err != nil {
		t.Fatalf("Select: %v", err)
	}

	dur := 150 * time.Millisecond
	tv := unix.NsecToTimeval(int64(dur))
	start := time.Now()
	err = unix.Select(0, nil, nil, nil, &tv)
	took := time.Since(start)
	if err != nil {
		t.Fatalf("Select: %v", err)
	}

	if took < dur {
		t.Errorf("Select: timeout should have been at least %v, got %v", dur, took)
	}
}

func TestStatvfs(t *testing.T) {
	if err := unix.Statvfs("", nil); err == nil {
		t.Fatal(`Statvfs("") expected failure`)
	}

	statvfs := unix.Statvfs_t{}
	if err := unix.Statvfs("/", &statvfs); err != nil {
		t.Errorf(`Statvfs("/") failed: %v`, err)
	}

	if t.Failed() {
		mount, err := exec.Command("mount").CombinedOutput()
		if err != nil {
			t.Logf("mount: %v\n%s", err, mount)
		} else {
			t.Logf("mount: %s", mount)
		}
	}
}
