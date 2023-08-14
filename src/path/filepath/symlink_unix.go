// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !windows && !plan9

package filepath

func evalSymlinks(path string) (string, error) {
	return walkSymlinks(path)
}
