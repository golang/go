// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

func F() bool {
	{
		x := false
		_ = x
	}
	if false {
		_ = func(x bool) {}
	}
	x := true
	return x
}

func G() func() bool {
	x := true
	return func() bool {
		{
			x := false
			_ = x
		}
		return x
	}
}
