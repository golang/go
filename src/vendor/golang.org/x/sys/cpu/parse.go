// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

import "strconv"

// parseRelease parses a dot-separated version number from the prefix
// of rel. It returns ok=true only if at least the major and minor
// components were successfully parsed; the patch component is
// best-effort. Trailing vendor or build suffixes such as
// "-generic", "+", "_hi3535", or "-rc1" are ignored.
//
// This is a copy of the Go runtime's parseRelease from
// https://golang.org/cl/209597, updated in https://golang.org/cl/781800.
func parseRelease(rel string) (major, minor, patch int, ok bool) {
	// next consumes a run of decimal digits from the front of rel,
	// returning the parsed value. If the digits are followed by a
	// '.', it is consumed and more is set so the caller knows to
	// parse another component; otherwise scanning terminates and
	// the rest of rel is discarded.
	next := func() (n int, more, ok bool) {
		i := 0
		for i < len(rel) && rel[i] >= '0' && rel[i] <= '9' {
			i++
		}
		if i == 0 {
			return 0, false, false
		}
		n, err := strconv.Atoi(rel[:i])
		if err != nil {
			return 0, false, false
		}
		if i < len(rel) && rel[i] == '.' {
			rel = rel[i+1:]
			return n, true, true
		}
		rel = ""
		return n, false, true
	}

	var more bool
	if major, more, ok = next(); !ok || !more {
		return 0, 0, 0, false
	}
	if minor, more, ok = next(); !ok {
		return 0, 0, 0, false
	}
	if !more {
		return major, minor, 0, true
	}
	patch, _, _ = next()
	return major, minor, patch, true
}
