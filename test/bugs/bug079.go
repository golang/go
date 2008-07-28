// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// $G $D/$F.go && echo BUG: succeeds incorrectly

package main

func f(int);  // parameter must be named
func g(int, int);  // parameter must be named

/* We had this discussion before and agreed that all parameters must be named. */
