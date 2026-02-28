// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js && wasm

package bytes_test

import (
	"bytes"
	"testing"
)

func TestIssue65571(t *testing.T) {
	b := make([]byte, 1<<31+1)
	b[1<<31] = 1
	i := bytes.IndexByte(b, 1)
	if i != 1<<31 {
		t.Errorf("IndexByte(b, 1) = %d; want %d", i, 1<<31)
	}
}
