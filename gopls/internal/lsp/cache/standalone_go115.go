// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.16
// +build !go1.16

package cache

// isStandaloneFile returns false, as the 'standaloneTags' setting is
// unsupported on Go 1.15 and earlier.
func isStandaloneFile(src []byte, standaloneTags []string) bool {
	return false
}
