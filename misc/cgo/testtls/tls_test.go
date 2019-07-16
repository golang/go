// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows

package cgotlstest

import "testing"

func TestTLS(t *testing.T) {
	testTLS(t)
}
