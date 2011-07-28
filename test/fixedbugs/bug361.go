// $G $D/$F.go || echo BUG: bug360

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 1908
// unreasonable width used to be internal fatal error

package test

func main() {
	buf := [1<<30]byte{}
	_ = buf[:]
}
