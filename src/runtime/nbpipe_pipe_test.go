// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin

package runtime_test

import (
	"runtime"
	"syscall"
	"testing"
)

func TestSetNonblock(t *testing.T) {
	t.Parallel()

	r, w, errno := runtime.Pipe()
	if errno != 0 {
		t.Fatal(syscall.Errno(errno))
	}
	defer func() {
		runtime.Close(r)
		runtime.Close(w)
	}()

	checkIsPipe(t, r, w)

	runtime.SetNonblock(r)
	runtime.SetNonblock(w)
	checkNonblocking(t, r, "reader")
	checkNonblocking(t, w, "writer")

	runtime.Closeonexec(r)
	runtime.Closeonexec(w)
	checkCloseonexec(t, r, "reader")
	checkCloseonexec(t, w, "writer")
}
