// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package os_test

import (
	"internal/testenv"
	. "os"
	"testing"
)

func TestErrProcessDone(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	p, err := StartProcess(testenv.GoToolPath(t), []string{"go"}, &ProcAttr{})
	if err != nil {
		t.Errorf("starting test process: %v", err)
	}
	p.Wait()
	if got := p.Signal(Kill); got != ErrProcessDone {
		t.Errorf("got %v want %v", got, ErrProcessDone)
	}
}
