// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package build provides tools for building Go packages.
package build

import "errors"

// ArchChar returns the architecture character for the given goarch.
// For example, ArchChar("amd64") returns "6".
func ArchChar(goarch string) (string, error) {
	switch goarch {
	case "386":
		return "8", nil
	case "amd64":
		return "6", nil
	case "arm":
		return "5", nil
	}
	return "", errors.New("unsupported GOARCH " + goarch)
}
