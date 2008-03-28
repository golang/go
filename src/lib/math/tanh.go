// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tanh

import		sinh "sinh"
export		tanh

/*
	tanh(arg) computes the hyperbolic tangent of its floating
	point argument.

	sinh and cosh are called except for large arguments, which
	would cause overflow improperly.
 */

func
tanh(arg double) double
{
	if arg < 0 {
		arg = -arg;
		if arg > 21 {
			return -1;
		}
		return -sinh.sinh(arg)/sinh.cosh(arg);
	}
	if arg > 21 {
		return 1;
	}
	return sinh.sinh(arg)/sinh.cosh(arg);
}
