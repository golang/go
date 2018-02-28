// compile

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 7346 : internal error "doasm" error due to checknil
// of a nil literal.

package main

func main() {
	_ = *(*int)(nil)
}
