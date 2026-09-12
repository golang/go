// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit

package jit

import (
	"testing"
	"unsafe"
)

type zeroSizedTailBase struct {
	prefix byte
	tail   [0]struct{}
}

type zeroSizedRepeatedTail struct {
	prefix byte
	tail   [3]struct{}
}

func TestTailTypeForZeroSizedElementsPreservesTrailingPadding(t *testing.T) {
	got := TailTypeFor[zeroSizedTailBase](3).runtimeType.Size_
	want := unsafe.Sizeof(zeroSizedRepeatedTail{})
	if got != want {
		t.Fatalf("repeated zero-sized tail size = %d, want %d", got, want)
	}
}
