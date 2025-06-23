// errorcheck

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// https://golang.org/issue/808

package main

type A [...]int	// ERROR "outside of array literal|invalid use of \[\.\.\.\]"


