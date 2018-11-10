// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lib

type I interface {
	m() string
}

type T struct{}

// m is not accessible from outside this package.
func (t *T) m() string {
	return "lib.T.m"
}
