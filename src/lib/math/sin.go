// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sin

import	sys "sys"
export	sin, cos

const
(
	p0	=  .1357884097877375669092680e8;
	p1	= -.4942908100902844161158627e7;
	p2	=  .4401030535375266501944918e6;
	p3	= -.1384727249982452873054457e5;
	p4	=  .1459688406665768722226959e3;
	q0	=  .8644558652922534429915149e7;
	q1	=  .4081792252343299749395779e6;
	q2	=  .9463096101538208180571257e4;
	q3	=  .1326534908786136358911494e3;
        piu2	=  .6366197723675813430755350e0;	// 2/pi
)

func
sinus(arg double, quad int) double
{
	var e, f, ysq, x, y, temp1, temp2 double;
	var k long;

	x = arg;
	if(x < 0) {
		x = -x;
		quad = quad+2;
	}
	x = x * piu2;	/* underflow? */
	if x > 32764 {
		e,y = sys.modf(x);
		e = e + double(quad);
		temp1,f = sys.modf(0.25*e);
		quad = int(e - 4*f);
	} else {
		k = long(x);
		y = x - double(k);
		quad = (quad + k) & 3;
	}

	if quad&1 != 0 {
		y = 1-y;
	}
	if quad > 1 {
		y = -y;
	}

	ysq = y*y;
	temp1 = ((((p4*ysq+p3)*ysq+p2)*ysq+p1)*ysq+p0)*y;
	temp2 = ((((ysq+q3)*ysq+q2)*ysq+q1)*ysq+q0);
	return temp1/temp2;
}

func
cos(arg double) double
{
	if arg < 0 {
		arg = -arg;
	}
	return sinus(arg, 1);
}

func
sin(arg double) double
{
	return sinus(arg, 0);
}
