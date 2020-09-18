// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type T struct { x int }

func F() interface{} {
	return [2]T{}
}

func P() interface{} {
	return &[2]T{}
}
