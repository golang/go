// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"errors"
	"testing"
)

func TestWasmErrorCodes(t *testing.T) {
	errmsg = nil
	if got := WasmError(nil); got != 0 {
		t.Fatalf("WasmError(nil) = %d, want 0", got)
	}
	if len(errmsg) != 0 {
		t.Fatalf("WasmError(nil) recorded %q", errmsg)
	}
	if got := WasmError(errors.New("bad input: 3 bytes")); got != 1 {
		t.Fatalf("WasmError(err) = %d, want 1", got)
	}
	if got := string(errmsg); got != "bad input: 3 bytes" {
		t.Fatalf("recorded %q, want the exact error text", got)
	}
}

func TestWasmErrorKeepsLastMessage(t *testing.T) {
	WasmError(errors.New("first"))
	WasmError(nil)
	if got := string(errmsg); got != "first" {
		t.Fatalf("WasmError(nil) disturbed the recorded text: %q", got)
	}
	WasmError(errors.New("second, longer text"))
	if got := string(errmsg); got != "second, longer text" {
		t.Fatalf("recorded %q after a second error", got)
	}
}
