// +build !amd64 !linux

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

import "testing"

func test24206(t *testing.T) {
	t.Skip("Skipping on non-amd64 or non-linux system")
}
