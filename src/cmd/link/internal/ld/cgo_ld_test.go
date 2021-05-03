// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
// +build cgo

package ld

import (
	"fmt"
	"runtime"
	"testing"
)

func TestWindowsBuildmodeCSharedASLR(t *testing.T) {
	platform := fmt.Sprintf("%s/%s", runtime.GOOS, runtime.GOARCH)
	switch platform {
	case "windows/amd64", "windows/386":
	default:
		t.Skip("skipping windows amd64/386 only test")
	}

	t.Run("aslr", func(t *testing.T) {
		testWindowsBuildmodeCSharedASLR(t, true)
	})
	t.Run("no-aslr", func(t *testing.T) {
		testWindowsBuildmodeCSharedASLR(t, false)
	})
}
