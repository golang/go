// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gover

import (
	"bytes"
	"strings"
)

var nl = []byte("\n")

// GoModLookup takes go.mod or go.work content,
// finds the first line in the file starting with the given key,
// and returns the value associated with that key.
//
// Lookup should only be used with non-factored verbs
// such as "go" and "toolchain", usually to find versions
// or version-like strings.
func GoModLookup(gomod []byte, key string) string {
	for len(gomod) > 0 {
		var line []byte
		line, gomod, _ = bytes.Cut(gomod, nl)
		line = bytes.TrimSpace(line)
		if v, ok := parseKey(line, key); ok {
			return v
		}
	}
	return ""
}

func parseKey(line []byte, key string) (string, bool) {
	if !strings.HasPrefix(string(line), key) {
		return "", false
	}
	s := strings.TrimPrefix(string(line), key)
	if len(s) == 0 || (s[0] != ' ' && s[0] != '\t') {
		return "", false
	}
	s, _, _ = strings.Cut(s, "//") // strip comments
	return strings.TrimSpace(s), true
}
