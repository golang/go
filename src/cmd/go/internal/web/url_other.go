// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !windows

package web

import (
	"errors"
	"path/filepath"
	"strings"
)

func convertFileURLPath(host, path string) (string, error) {
	if host != "" && strings.EqualFold(host, "localhost") {
		return "", errors.New("file URL specifies non-local host")
	}
	return filepath.FromSlash(path), nil
}
