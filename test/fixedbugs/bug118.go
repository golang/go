// $G $D/$F.go || echo BUG should compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func Send(c chan int) int {
	select {
	default:
		return 1;
	}
	return 2;
}
