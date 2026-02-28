// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// https://gcc.gnu.org/PR101994
// gccgo compiler crash with zero-sized result.

package p

type Empty struct{}

func F() (int, Empty) {
	return 0, Empty{}
}
