// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris

package os_test

import (
	"internal/testenv"
	. "os"
	"testing"
)

func TestErrProcessDone(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	path, err := testenv.GoTool()
	if err != nil {
		t.Errorf("finding go tool: %v", err)
	}
	p, err := StartProcess(path, []string{"go"}, &ProcAttr{})
	if err != nil {
		t.Errorf("starting test process: %v", err)
	}
	p.Wait()
	if got := p.Signal(Kill); got != ErrProcessDone {
		t.Errorf("got %v want %v", got, ErrProcessDone)
	}
}
