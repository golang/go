// compile

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// http://code.google.com/p/go/issues/detail?id=806
// triggered out of registers on 8g

package bug283

type Point struct {
	x int
	y int
}

func dist(p0, p1 Point) float64 {
	return float64((p0.x-p1.x)*(p0.x-p1.x) + (p0.y-p1.y)*(p0.y-p1.y))
}
