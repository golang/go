// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"


void
dump(byte *p, int32 n)
{
	int32 i;

	for(i=0; i<n; i++) {
		sys·printpointer((byte*)(p[i]>>4));
		sys·printpointer((byte*)(p[i]&0xf));
		if((i&15) == 15)
			prints("\n");
		else
			prints(" ");
	}
	if(n & 15)
		prints("\n");
}

void
prints(int8 *s)
{
	sys·write(1, s, findnull(s));
}

void
sys·printpc(void *p)
{
	prints("PC=0x");
	sys·printpointer(sys·getcallerpc(p));
}

void
sys·printbool(bool v)
{
	if(v) {
		sys·write(1, (byte*)"true", 4);
		return;
	}
	sys·write(1, (byte*)"false", 5);
}

void
sys·printfloat(float64 v)
{
	byte buf[20];
	int32 e, s, i, n;
	float64 h;

	if(isNaN(v)) {
		sys·write(1, "NaN", 3);
		return;
	}
	if(isInf(v, 0)) {
		sys·write(1, "+Inf", 4);
		return;
	}
	if(isInf(v, -1)) {
		sys·write(1, "+Inf", 4);
		return;
	}


	n = 7;	// digits printed
	e = 0;	// exp
	s = 0;	// sign
	if(v != 0) {
		// sign
		if(v < 0) {
			v = -v;
			s = 1;
		}

		// normalize
		while(v >= 10) {
			e++;
			v /= 10;
		}
		while(v < 1) {
			e--;
			v *= 10;
		}

		// round
		h = 5;
		for(i=0; i<n; i++)
			h /= 10;
		v += h;
		if(v >= 10) {
			e++;
			v /= 10;
		}
	}

	// format +d.dddd+edd
	buf[0] = '+';
	if(s)
		buf[0] = '-';
	for(i=0; i<n; i++) {
		s = v;
		buf[i+2] = s+'0';
		v -= s;
		v *= 10.;
	}
	buf[1] = buf[2];
	buf[2] = '.';

	buf[n+2] = 'e';
	buf[n+3] = '+';
	if(e < 0) {
		e = -e;
		buf[n+3] = '-';
	}

	buf[n+4] = (e/100) + '0';
	buf[n+5] = (e/10)%10 + '0';
	buf[n+6] = (e%10) + '0';
	sys·write(1, buf, n+7);
}

void
sys·printuint(uint64 v)
{
	byte buf[100];
	int32 i;

	for(i=nelem(buf)-1; i>0; i--) {
		buf[i] = v%10 + '0';
		if(v < 10)
			break;
		v = v/10;
	}
	sys·write(1, buf+i, nelem(buf)-i);
}

void
sys·printint(int64 v)
{
	if(v < 0) {
		sys·write(1, "-", 1);
		v = -v;
	}
	sys·printuint(v);
}

void
sys·printpointer(void *p)
{
	uint64 v;
	byte buf[100];
	int32 i;

	v = (int64)p;
	for(i=nelem(buf)-1; i>0; i--) {
		buf[i] = v%16 + '0';
		if(buf[i] > '9')
			buf[i] += 'a'-'0'-10;
		if(v < 16)
			break;
		v = v/16;
	}
	sys·write(1, buf+i, nelem(buf)-i);
}

void
sys·printstring(string v)
{
	if(v != nil)
		sys·write(1, v->str, v->len);
}

void
sys·printsp(void)
{
	sys·write(1, " ", 1);
}

void
sys·printnl(void)
{
	sys·write(1, "\n", 1);
}
