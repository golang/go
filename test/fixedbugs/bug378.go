// run

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2497

package main

type Header struct{}
func (h Header) Method() {}

var _ interface{} = Header{}

func main() {
  	type X Header
  	var _ interface{} = X{}
}
