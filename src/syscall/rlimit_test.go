// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package syscall_test

import (
	"os"
	"runtime"
	"testing"
)

func TestOpenFileLimit(t *testing.T) {
	// For open file count,
	// macOS sets the default soft limit to 256 and no hard limit.
	// CentOS and Fedora set the default soft limit to 1024,
	// with hard limits of 4096 and 524288, respectively.
	// Check that we can open 1200 files, which proves
	// that the rlimit is being raised appropriately on those systems.
	fileCount := 1200

	// OpenBSD has a default soft limit of 512 and hard limit of 1024.
	if runtime.GOOS == "openbsd" {
		fileCount = 768
	}

	var files []*os.File
	for i := 0; i < fileCount; i++ {
		f, err := os.Open("rlimit.go")
		if err != nil {
			t.Error(err)
			break
		}
		files = append(files, f)
	}

	for _, f := range files {
		f.Close()
	}
}
