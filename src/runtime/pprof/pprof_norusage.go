// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !darwin,!linux

package pprof

import (
	"io"
)

// Stub call for platforms that don't support rusage.
func addMaxRSS(w io.Writer) {
}
