// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

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
	tg.creatingTemp("x")
	tg.run("build", tg.path("x.go"))
	fi, err := os.Stat("x")
	if err != nil {
		t.Fatal(err)
	}
	if mode := fi.Mode(); mode&0077 != 0 {
		t.Fatalf("wrote x with mode=%v, wanted no 0077 bits", mode)
	}
}
