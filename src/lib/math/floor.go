// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

export	floor, ceil

/*
 * floor and ceil-- greatest integer <= arg
 * (resp least >=)
 */

func
floor(arg float64) float64
{
	var fract, d float64;

	d = arg;
	if d < 0 {
		d,fract = sys.modf(-d);
		if fract != 0.0 {
			d = d+1;
		}
		d = -d;
	} else {
		d,fract = sys.modf(d);
	}
	return d;
}

func
ceil(arg float64) float64
{
	return -floor(-arg);
}
