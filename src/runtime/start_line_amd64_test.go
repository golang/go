// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime/internal/startlinetest"
	"testing"
)

// TestStartLineAsm tests the start line metadata of an assembly function. This
// is only tested on amd64 to avoid the need for a proliferation of per-arch
// copies of this function.
func TestStartLineAsm(t *testing.T) {
	startlinetest.CallerStartLine = callerStartLine

	const wantLine = 23
	got := startlinetest.AsmFunc()
	if got != wantLine {
		t.Errorf("start line got %d want %d", got, wantLine)
	}
}
