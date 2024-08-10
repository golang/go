// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"internal/poll"
	. "os"
	"testing"
)

var (
	copyFileTests = []copyFileTestFunc{newCopyFileRangeTest}
	copyFileHooks = []copyFileTestHook{hookCopyFileRange}
)

func testCopyFiles(t *testing.T, size, limit int64) {
	testCopyFileRange(t, size, limit)
}

func testCopyFileRange(t *testing.T, size int64, limit int64) {
	dst, src, data, hook, name := newCopyFileRangeTest(t, size)
	testCopyFile(t, dst, src, data, hook, limit, name)
}

// newCopyFileRangeTest initializes a new test for copy_file_range.
// It hooks package os' call to poll.CopyFileRange and returns the hook,
// so it can be inspected.
func newCopyFileRangeTest(t *testing.T, size int64) (dst, src *File, data []byte, hook *copyFileHook, name string) {
	t.Helper()

	name = "newCopyFileRangeTest"

	dst, src, data = newCopyFileTest(t, size)
	hook, _ = hookCopyFileRange(t)

	return
}

func hookCopyFileRange(t *testing.T) (hook *copyFileHook, name string) {
	name = "hookCopyFileRange"

	hook = new(copyFileHook)
	orig := *PollCopyFileRangeP
	t.Cleanup(func() {
		*PollCopyFileRangeP = orig
	})
	*PollCopyFileRangeP = func(dst, src *poll.FD, remain int64) (int64, bool, error) {
		hook.called = true
		hook.dstfd = dst.Sysfd
		hook.srcfd = src.Sysfd
		hook.written, hook.handled, hook.err = orig(dst, src, remain)
		return hook.written, hook.handled, hook.err
	}
	return
}
