// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !windows

package upki

import "os"

func openCacheFile(path string) (*os.File, error) {
	return os.Open(path)
}

func readFilterFile(path string) ([]byte, error) {
	return os.ReadFile(path)
}
