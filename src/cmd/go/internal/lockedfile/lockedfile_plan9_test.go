// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build plan9

package lockedfile

import (
	"errors"
	"testing"
)

func TestIsLocked(t *testing.T) {
	for _, msg := range []string{
		"open /x: file is locked",                  // cwfs, kfs
		"open /x: exclusive lock",                  // fossil
		"open /x: exclusive use file already open", // ramfs
		"open /x: file locked",                     // hjfs
	} {
		if !isLocked(errors.New(msg)) {
			t.Errorf("isLocked(%q) = false, want true", msg)
		}
	}
	if isLocked(errors.New("open /x: file does not exist")) {
		t.Errorf("isLocked(non-lock error) = true, want false")
	}
}
