// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"crypto/sha256"
	"runtime"
	"testing"
)

func Verify(token, salt string) [32]byte {
	return sha256.Sum256([]byte(token + salt))
}

func TestIssue71943(t *testing.T) {
	if n := testing.AllocsPerRun(10, func() {
		runtime.KeepAlive(Verify("teststring", "test"))
	}); n > 0 {
		t.Fatalf("unexpected allocation: %f", n)
	}
}
