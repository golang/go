// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

import		math "math"

/*
 *	tanh(arg) computes the hyperbolic tangent of its floating
 *	point argument.
 *
 *	sinh and cosh are called except for large arguments, which
 *	would cause overflow improperly.
 */

export func
tanh(arg float64) float64
{
	if arg < 0 {
		arg = -arg;
		if arg > 21 {
			return -1;
		}
		return -sinh(arg)/cosh(arg);
	}
	if arg > 21 {
		return 1;
	}
	return sinh(arg)/cosh(arg);
}
