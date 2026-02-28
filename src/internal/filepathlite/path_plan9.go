// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filepathlite

import (
	"internal/bytealg"
	"internal/stringslite"
)

const (
	Separator     = '/'    // OS-specific path separator
	ListSeparator = '\000' // OS-specific path list separator
)

func IsPathSeparator(c uint8) bool {
	return Separator == c
}

func isLocal(path string) bool {
	return unixIsLocal(path)
}

func localize(path string) (string, error) {
	if path[0] == '#' || bytealg.IndexByteString(path, 0) >= 0 {
		return "", errInvalidPath
	}
	return path, nil
}

// IsAbs reports whether the path is absolute.
func IsAbs(path string) bool {
	return stringslite.HasPrefix(path, "/") || stringslite.HasPrefix(path, "#")
}

// volumeNameLen returns length of the leading volume name on Windows.
// It returns 0 elsewhere.
func volumeNameLen(path string) int {
	return 0
}
