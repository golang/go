// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"os"
	"testing"
)

func TestCheckPidfd(t *testing.T) {
	// This doesn't test anything, but merely allows to check that pidfd
	// is working (and thus being tested) in CI on some platforms.
	if err := os.CheckPidfdOnce(); err != nil {
		t.Log("checkPidfd:", err)
	} else {
		t.Log("pidfd syscalls work")
	}
}
