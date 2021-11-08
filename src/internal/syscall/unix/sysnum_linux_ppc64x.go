// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64 || ppc64le

package unix

const (
	getrandomTrap     uintptr = 359
	copyFileRangeTrap uintptr = 379
)
