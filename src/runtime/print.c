// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"


void
dump(byte *p, int32 n)
{
	uint32 v;
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
	sys·write(1, "printfloat", 10);
}

void
sys·printint(int64 v)
{
	byte buf[100];
	int32 i, s;

	s = 0;
	if(v < 0) {
		v = -v;
		s = 1;
		if(v < 0) {
			sys·write(1, (byte*)"-oo", 3);
			return;
		}
	}

	for(i=nelem(buf)-1; i>0; i--) {
		buf[i] = v%10 + '0';
		if(v < 10)
			break;
		v = v/10;
	}
	if(s) {
		i--;
		buf[i] = '-';
	}
	sys·write(1, buf+i, nelem(buf)-i);
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
