// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

import		math "floor"
import		math "sqrt"
import		math "log"
import		math "exp"
export		pow

/*
	arg1 ^ arg2 (exponentiation)
 */

func
pow(arg1,arg2 double) double
{
	var temp double;
	var l long;

	if arg2 < 0 {
		return 1/pow(arg1, -arg2);
	}
	if arg1 <= 0 {
		if(arg1 == 0) {
			if arg2 <= 0 {
				panic "return sys.NaN()";
			}
			return 0;
		}

		temp = floor(arg2);
		if temp != arg2 {
			panic "return sys.NaN()";
		}

		l = long(temp);
		if l&1 != 0 {
			return -pow(-arg1, arg2);
		}
		return pow(-arg1, arg2);
	}

	temp = floor(arg2);
	if temp != arg2 {
		if arg2-temp == .5 {
			if temp == 0 {
				return sqrt(arg1);
			}
			return pow(arg1, temp) * sqrt(arg1);
		}
		return exp(arg2 * log(arg1));
	}

	l = long(temp);
	temp = 1;
	for {
		if l&1 != 0 {
			temp = temp*arg1;
		}
		l = l>>1;
		if l == 0 {
			return temp;
		}
		arg1 = arg1*arg1;
	}
}
