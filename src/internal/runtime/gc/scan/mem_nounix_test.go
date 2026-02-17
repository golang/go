// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !unix

package scan_test

import (
	"testing"
)

func makeMem(t testing.TB, nPages int) ([]uintptr, func()) {
	t.Skip("mmap unsupported")
	return nil, nil
}
