// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 20145: some func types weren't dowidth-ed by the front end,
// leading to races in the backend.

package p

func f() {
	_ = (func())(nil)
}
