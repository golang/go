// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.21

package countertest

import "testing"

func init() {
	// Extra safety check for go1.21+.
	if !testing.Testing() {
		panic("use of this package is disallowed in non-testing code")
	}
}
