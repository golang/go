// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package runtime

const canCreateFile = true

// create returns an fd to a write-only file.
func create(name *byte, perm int32) int32 {
	return open(name, _O_CREAT|_O_WRONLY|_O_TRUNC, perm)
}
