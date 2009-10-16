// $G $D/$F.go || echo BUG: should compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T struct {
	children []T;
}

/*
uetli:/home/gri/go/test/bugs gri$ 6g bug210.go
bug210.go:10: invalid recursive type []T
*/
