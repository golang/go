// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

/*
	Coefficients are #3370 from Hart & Cheney (18.80D).
*/
const
(
	sp0	=  .1357884097877375669092680e8;
	sp1	= -.4942908100902844161158627e7;
	sp2	=  .4401030535375266501944918e6;
	sp3	= -.1384727249982452873054457e5;
	sp4	=  .1459688406665768722226959e3;
	sq0	=  .8644558652922534429915149e7;
	sq1	=  .4081792252343299749395779e6;
	sq2	=  .9463096101538208180571257e4;
	sq3	=  .1326534908786136358911494e3;

	spiu2	=  .6366197723675813430755350e0;	// 2/pi
)

func sinus(arg float64, quad int) float64 {
	x := arg;
	if(x < 0) {
		x = -x;
		quad = quad+2;
	}
	x = x * spiu2;	/* underflow? */
	var y float64;
	if x > 32764 {
		var e float64;
		e, y = sys.modf(x);
		e = e + float64(quad);
		temsp1, f := sys.modf(0.25*e);
		quad = int(e - 4*f);
	} else {
		k := int32(x);
		y = x - float64(k);
		quad = (quad + int(k)) & 3;
	}

	if quad&1 != 0 {
		y = 1-y;
	}
	if quad > 1 {
		y = -y;
	}

	ysq := y*y;
	temsp1 := ((((sp4*ysq+sp3)*ysq+sp2)*ysq+sp1)*ysq+sp0)*y;
	temsp2 := ((((ysq+sq3)*ysq+sq2)*ysq+sq1)*ysq+sq0);
	return temsp1/temsp2;
}

export func Cos(arg float64) float64 {
	if arg < 0 {
		arg = -arg;
	}
	return sinus(arg, 1);
}

export func Sin(arg float64) float64 {
	return sinus(arg, 0);
}
