// compile

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Gccgo used to give an incorrect error
// bug495.go:16:2: error: missing statement after label

package p

func F(i int) {
	switch i {
	case 0:
		goto lab
	lab:
		fallthrough
	case 1:
	}
}
