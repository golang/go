// compile

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test case that gccgo failed to compile.

package p

func F() []string {
	return []string{""}
}

var V = append(F())
