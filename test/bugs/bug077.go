// $G $D/$F.go || echo BUG: should compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var exit int;
exit:  // this shouldn't be legal
}

/*
Within a scope, an identifier should have only one association - it cannot be
a variable and a label at the same time.
*/
