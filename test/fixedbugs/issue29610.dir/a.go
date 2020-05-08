// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type I interface {
	M(init bool)
}

var V I

func init() {
	V = nil
}
