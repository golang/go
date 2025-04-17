// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || openbsd

package syscall_test

import (
	"os/exec"
	"syscall"
	"testing"
)

const MNT_WAIT = 1
const MNT_NOWAIT = 2

func TestGetfsstat(t *testing.T) {
	const flags = MNT_NOWAIT // see Issue 16937
	n, err := syscall.Getfsstat(nil, flags)
	t.Logf("Getfsstat(nil, %d) = (%v, %v)", flags, n, err)
	if err != nil {
		t.Fatal(err)
	}

	data := make([]syscall.Statfs_t, n)
	n2, err := syscall.Getfsstat(data, flags)
	t.Logf("Getfsstat([]syscall.Statfs_t, %d) = (%v, %v)", flags, n2, err)
	if err != nil {
		t.Fatal(err)
	}
	if n != n2 {
		t.Errorf("Getfsstat(nil) = %d, but subsequent Getfsstat(slice) = %d", n, n2)
	}
	for i, stat := range data {
		if stat == (syscall.Statfs_t{}) {
			t.Errorf("index %v is an empty Statfs_t struct", i)
		}
	}
	if t.Failed() {
		for i, stat := range data[:n2] {
			t.Logf("data[%v] = %+v", i, stat)
		}
		mount, err := exec.Command("mount").CombinedOutput()
		if err != nil {
			t.Logf("mount: %v\n%s", err, mount)
		} else {
			t.Logf("mount: %s", mount)
		}
	}
}
