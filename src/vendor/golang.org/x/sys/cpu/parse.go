// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

import "strconv"

// parseRelease parses a dot-separated version number. It follows the semver
// syntax, but allows the minor and patch versions to be elided.
//
// This is a copy of the Go runtime's parseRelease from
// https://golang.org/cl/209597.
func parseRelease(rel string) (major, minor, patch int, ok bool) {
	// Strip anything after a dash or plus.
	for i := range len(rel) {
		if rel[i] == '-' || rel[i] == '+' {
			rel = rel[:i]
			break
		}
	}

	next := func() (int, bool) {
		for i := range len(rel) {
			if rel[i] == '.' {
				ver, err := strconv.Atoi(rel[:i])
				rel = rel[i+1:]
				return ver, err == nil
			}
		}
		ver, err := strconv.Atoi(rel)
		rel = ""
		return ver, err == nil
	}
	if major, ok = next(); !ok || rel == "" {
		return
	}
	if minor, ok = next(); !ok || rel == "" {
		return
	}
	patch, ok = next()
	return
}
