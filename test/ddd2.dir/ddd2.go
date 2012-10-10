// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is compiled and then imported by ddd3.go.

package ddd

func Sum(args ...int) int {
	s := 0
	for _, v := range args {
		s += v
	}
	return s
}

