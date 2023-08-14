// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package safefilepath manipulates operating-system file paths.
package safefilepath

import (
	"errors"
)

var errInvalidPath = errors.New("invalid path")

// FromFS converts a slash-separated path into an operating-system path.
//
// FromFS returns an error if the path cannot be represented by the operating
// system. For example, paths containing '\' and ':' characters are rejected
// on Windows.
func FromFS(path string) (string, error) {
	return fromFS(path)
}
