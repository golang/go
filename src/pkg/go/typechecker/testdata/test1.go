// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// const and var declarations

package P1

const (
	c1         /* ERROR "missing initializer" */
	c2     int = 0
	c3, c4 = 0
)
