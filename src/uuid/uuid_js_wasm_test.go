// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js && wasm

package uuid

import (
	"testing"
	"time"
)

func TestNewV7SubmillisecondBitsOnWasm(t *testing.T) {
	allZero := true
	for range 20 {
		u := NewV7()
		if u[6]&0x0f != 0 || u[7] != 0 {
			allZero = false
			break
		}
		time.Sleep(2 * time.Millisecond)
	}
	if allZero {
		t.Fatal("NewV7 sub-millisecond bits are always zero on js/wasm")
	}
}
