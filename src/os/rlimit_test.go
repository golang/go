// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	. "os"
	"runtime"
	"testing"
)

func TestOpenFileLimit(t *testing.T) {
	if runtime.GOOS == "openbsd" && (runtime.GOARCH == "arm" || runtime.GOARCH == "arm64") {
		t.Skip("broken on openbsd/arm and openbsd/arm64 builder - go.dev/issue/51713")
	}

	// For open file count,
	// macOS sets the default soft limit to 256 and no hard limit.
	// CentOS and Fedora set the default soft limit to 1024,
	// with hard limits of 4096 and 524288, respectively.
	// Check that we can open 1200 files, which proves
	// that the rlimit is being raised appropriately on those systems.
	var files []*File
	for i := 0; i < 1200; i++ {
		f, err := Open("rlimit.go")
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
