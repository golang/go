// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo

package main

import (
	"runtime"
	"testing"
)

func canInternalLink() bool {
	switch runtime.GOOS {
	case "aix":
		return false
	case "dragonfly":
		return false
	case "freebsd":
		switch runtime.GOARCH {
		case "arm64":
			return false
		}
	case "linux":
		switch runtime.GOARCH {
		case "arm64", "mips64", "mips64le", "mips", "mipsle", "ppc64", "ppc64le":
			return false
		}
	case "openbsd":
		switch runtime.GOARCH {
		case "arm64":
			return false
		}
	}
	return true
}

func TestInternalLinkerCgoExec(t *testing.T) {
	if !canInternalLink() {
		t.Skip("skipping; internal linking is not supported")
	}
	testGoExec(t, true, false)
}

func TestExternalLinkerCgoExec(t *testing.T) {
	testGoExec(t, true, true)
}

func TestCgoLib(t *testing.T) {
	testGoLib(t, true)
}
