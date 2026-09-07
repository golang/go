// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package goarch_test

import (
	"internal/goarch"
	"testing"
	"unsafe"
)

func TestInt64Align(t *testing.T) {
	if got, want := uintptr(goarch.Int64Align), unsafe.Alignof(int64(0)); got != want {
		t.Fatalf("Int64Align = %d, want %d", got, want)
	}
}
