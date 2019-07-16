// compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package P

const a = 0;

func f(a int) {
	a = 0;
}

/*
bug161.go:8: operation LITERAL not allowed in assignment context
*/
