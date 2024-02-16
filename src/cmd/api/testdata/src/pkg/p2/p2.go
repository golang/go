// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p2

type Twoer interface {
	// Deprecated: No good.
	PackageTwoMeth()
}

// Deprecated: No good.
func F() string {}

func G() Twoer {}

func NewError(s string) error {}
