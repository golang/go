// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package govulncheck

import (
	"path/filepath"
	"strings"
)

// AbsRelShorter takes path and returns its path relative
// to the current directory, if shorter. Returns path
// when path is an empty string or upon any error.
func AbsRelShorter(path string) string {
	if path == "" {
		return ""
	}

	c, err := filepath.Abs(".")
	if err != nil {
		return path
	}
	r, err := filepath.Rel(c, path)
	if err != nil {
		return path
	}

	rSegments := strings.Split(r, string(filepath.Separator))
	pathSegments := strings.Split(path, string(filepath.Separator))
	if len(rSegments) < len(pathSegments) {
		return r
	}
	return path
}
