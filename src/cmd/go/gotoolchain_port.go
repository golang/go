// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"strings"
)

var (
	nl           = []byte("\n")
	comment      = []byte("//")
	goKey        = []byte("go")
	toolchainKey = []byte("toolchain")
)

// parseKey checks whether line begings with key ("go" or "toolchain").
// If so, it returns the remainder of the line (the argument).
func parseKey(line, key []byte) string {
	if !bytes.HasPrefix(line, key) {
		return ""
	}
	line = bytes.TrimPrefix(line, key)
	if len(line) == 0 || (line[0] != ' ' && line[0] != '\t') {
		return ""
	}
	line, _, _ = bytes.Cut(line, comment) // strip comments
	return string(bytes.TrimSpace(line))
}

// toolchainMax returns the max of x and y as toolchain names
// like go1.19.4, comparing the versions.
func toolchainMax(x, y string) string {
	if toolchainCmp(x, y) >= 0 {
		return x
	}
	return y
}

// toolchainCmp returns -1, 0, or +1 depending on whether
// x < y, x == y, or x > y, interpreted as toolchain versions.
func toolchainCmp(x, y string) int {
	if x == y {
		return 0
	}
	if y == "" {
		return +1
	}
	if x == "" {
		return -1
	}
	if !strings.HasPrefix(x, "go1") && !strings.HasPrefix(y, "go1") {
		return 0
	}
	if !strings.HasPrefix(x, "go1") {
		return +1
	}
	if !strings.HasPrefix(y, "go1") {
		return -1
	}
	x = strings.TrimPrefix(x, "go")
	y = strings.TrimPrefix(y, "go")
	for x != "" || y != "" {
		if x == y {
			return 0
		}
		xN, xRest := versionCut(x)
		yN, yRest := versionCut(y)
		if xN > yN {
			return +1
		}
		if xN < yN {
			return -1
		}
		x = xRest
		y = yRest
	}
	return 0
}

// versionCut cuts the version x after the next dot or before the next non-digit,
// returning the leading decimal found and the remainder of the string.
func versionCut(x string) (int, string) {
	// Treat empty string as infinite source of .0.0.0...
	if x == "" {
		return 0, ""
	}
	i := 0
	v := 0
	for i < len(x) && '0' <= x[i] && x[i] <= '9' {
		v = v*10 + int(x[i]-'0')
		i++
	}
	// Treat non-empty non-number as -1 (for release candidates, etc),
	// but stop at next number.
	if i == 0 {
		for i < len(x) && (x[i] < '0' || '9' < x[i]) {
			i++
		}
		if i < len(x) && x[i] == '.' {
			i++
		}
		if strings.Contains(x[:i], "alpha") {
			return -3, x[i:]
		}
		if strings.Contains(x[:i], "beta") {
			return -2, x[i:]
		}
		return -1, x[i:]
	}
	if i < len(x) && x[i] == '.' {
		i++
	}
	return v, x[i:]
}
