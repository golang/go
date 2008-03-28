// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package floor

import	sys "sys"
export	floor, ceil

/*
 * floor and ceil-- greatest integer <= arg
 * (resp least >=)
 */

func
floor(arg double) double
{
	var fract, d double;

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
ceil(arg double) double
{
	return -floor(-arg);
}
