// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo

package main

import (
	"runtime"
	"testing"
)

func TestInternalLinkerCgoFile(t *testing.T) {
	if !canInternalLink() {
		t.Skip("skipping; internal linking is not supported")
	}
	testGoFile(t, true, false)
}

func canInternalLink() bool {
	switch runtime.GOOS {
	case "dragonfly":
		return false
	case "linux":
		switch runtime.GOARCH {
		case "arm64", "mips64", "mips64le", "mips", "mipsle":
			return false
		}
	}
	return true
}

func TestExternalLinkerCgoFile(t *testing.T) {
	testGoFile(t, true, true)
}
