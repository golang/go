// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !aix && !darwin && !dragonfly && !freebsd && !linux && !netbsd && !openbsd && !solaris && !windows

package pprof

import (
	"io"
)

// Stub call for platforms that don't support rusage.
func addMaxRSS(w io.Writer) {
}
