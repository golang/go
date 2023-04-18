// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 || !gc || purego

package bigmod

func montgomeryLoop(d, a, b, m []uint, m0inv uint) uint {
	return montgomeryLoopGeneric(d, a, b, m, m0inv)
}
