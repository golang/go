// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main


func main() {
	fired := false; _ = fired;
}
/*
bug9.go:5: defaultlit: unknown literal: LITERAL-B0 a(1)
bug9.go:5: fatal error: addvar: n=NAME-fired G0 a(1) l(5) t=<N> nil
*/
