// $G $D/$F.go || echo BUG should compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T struct { s string }
var t = T{"hi"}

func main() {}

/*
bug112.go:6: illegal conversion of constant to T
*/
