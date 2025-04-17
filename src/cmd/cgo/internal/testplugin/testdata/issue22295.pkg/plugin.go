// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var f *int

func init() {
	f = new(int)
	*f = 2503
}

func F() int { return *f }

func main() {}
