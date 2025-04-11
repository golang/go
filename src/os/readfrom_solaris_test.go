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
	copyFileTests = []copyFileTestFunc{newSendfileTest}
	copyFileHooks = []copyFileTestHook{hookSendFile}
)

func testCopyFiles(t *testing.T, size, limit int64) {
	testSendfile(t, size, limit)
}

func testSendfile(t *testing.T, size int64, limit int64) {
	dst, src, data, hook, name := newSendfileTest(t, size)
	testCopyFile(t, dst, src, data, hook, limit, name)
}

// newSendFileTest initializes a new test for sendfile over copy_file_range.
// It hooks package os' call to poll.SendFile and returns the hook,
// so it can be inspected.
func newSendfileTest(t *testing.T, size int64) (dst, src *File, data []byte, hook *copyFileHook, name string) {
	t.Helper()

	name = "newSendfileTest"

	dst, src, data = newCopyFileTest(t, size)
	hook, _ = hookSendFile(t)

	return
}

func hookSendFile(t *testing.T) (*copyFileHook, string) {
	return hookSendFileTB(t), "hookSendFile"
}

func hookSendFileTB(tb testing.TB) *copyFileHook {
	hook := new(copyFileHook)
	orig := poll.TestHookDidSendFile
	tb.Cleanup(func() {
		poll.TestHookDidSendFile = orig
	})
	poll.TestHookDidSendFile = func(dstFD *poll.FD, src uintptr, written int64, err error, handled bool) {
		hook.called = true
		hook.dstfd = dstFD.Sysfd
		hook.srcfd = int(src)
		hook.written = written
		hook.err = err
		hook.handled = handled
	}
	return hook
}
