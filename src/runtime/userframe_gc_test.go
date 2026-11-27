// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit

package runtime_test

import (
	"runtime"
	"testing"
)

func TestUserFramePersistentTree(t *testing.T) {
	if !runtime.UserFramePersistentTreeForTest() {
		t.Fatal("registry lost balance, ordering, or a retained immutable version")
	}
}

func TestUserFrameStableStackMapBlocks(t *testing.T) {
	if !runtime.UserFrameStableStackMapBlocksForTest() {
		t.Fatal("stack-map growth moved entries, lost copied masks, or invalidated an old root")
	}
}
