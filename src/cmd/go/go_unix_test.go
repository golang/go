// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package main_test

import (
	"os"
	"syscall"
	"testing"
)

func TestGoBuildUmask(t *testing.T) {
	// Do not use tg.parallel; avoid other tests seeing umask manipulation.
	mask := syscall.Umask(0077) // prohibit low bits
	defer syscall.Umask(mask)
	tg := testgo(t)
	defer tg.cleanup()
	tg.tempFile("x.go", `package main; func main() {}`)
	// Make sure artifact will be output to /tmp/... in case the user
	// has POSIX acl's on their go source tree.
	// See issue 17909.
	exe := tg.path("x")
	tg.creatingTemp(exe)
	tg.run("build", "-o", exe, tg.path("x.go"))
	fi, err := os.Stat(exe)
	if err != nil {
		t.Fatal(err)
	}
	if mode := fi.Mode(); mode&0077 != 0 {
		t.Fatalf("wrote x with mode=%v, wanted no 0077 bits", mode)
	}
}
