// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	type Slice []byte;
	a := [...]byte{ 0 };
	b := Slice(a[0:]);	// This should be OK.
	c := Slice(a);		// ERROR "invalid|illegal|cannot"
	_, _ = b, c;
}
