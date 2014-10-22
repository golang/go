// compile

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The gccgo compiler used to crash building a comparison between an
// interface and an empty struct literal.

package p
 
type S struct{}

func F(v interface{}) bool {
	return v == S{}
}
