// run

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// http://code.google.com/p/go/issues/detail?id=920

package main

type X struct { x []X }

func main() {
	type Y struct { x []Y }	// used to get invalid recursive type
}
