// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (amd64 || arm64 || wasm)

package simd_test

import (
	"simd/archsimd"
	"testing"
)

var int8x16Sink archsimd.Int8x16

func expectPanic(t *testing.T, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Error("did not panic")
		}
	}()
	f()
}

func TestArrayNilChecks(t *testing.T) {
	t.Run("load", func(t *testing.T) {
		expectPanic(t, func() {
			int8x16Sink = archsimd.LoadInt8x16Array(nil)
		})
	})
	t.Run("store", func(t *testing.T) {
		expectPanic(t, func() {
			var a [16]int8
			archsimd.LoadInt8x16Array(&a).StoreArray(nil)
		})
	})
}
