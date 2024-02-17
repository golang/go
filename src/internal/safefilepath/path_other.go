// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !windows

package safefilepath

import (
	"internal/bytealg"
	"runtime"
)

func fromFS(path string) (string, error) {
	if runtime.GOOS == "plan9" {
		if len(path) > 0 && path[0] == '#' {
			return "", errInvalidPath
		}
	}
	if bytealg.IndexByteString(path, 0) >= 0 {
		return "", errInvalidPath
	}
	return path, nil
}
