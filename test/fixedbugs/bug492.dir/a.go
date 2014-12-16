// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type s struct {
	s string
}

func F1(s s) {
}

func F2() s {
	return s{""}
}
