// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main
var v1 = ([10]int)(nil);	// ERROR "illegal|nil|invalid"
var v2 [10]int = nil;		// ERROR "illegal|nil|incompatible"
var v3 [10]int;
var v4 = nil;	// ERROR "nil"
func main() {
	v3 = nil;		// ERROR "illegal|nil|incompatible"
}
