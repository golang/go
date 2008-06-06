// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// $G $F.go && echo BUG: compilation succeeds incorrectly

package main

type Iputs	interface {
	puts func(s string);  // BUG no func allowed
}
