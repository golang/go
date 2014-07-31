// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "type.h"
#include "../../cmd/ld/textflag.h"

//static Lock debuglock;

static void vprintf(int8*, byte*);

// write to goroutine-local buffer if diverting output,
// or else standard error.
static void
gwrite(void *v, intgo n)
{
	if(g == nil || g->writebuf == nil) {
		runtime·write(2, v, n);
		return;
	}

	if(g->writenbuf == 0)
		return;

	if(n > g->writenbuf)
		n = g->writenbuf;
	runtime·memmove(g->writebuf, v, n);
	g->writebuf += n;
	g->writenbuf -= n;
}

void
runtime·dump(byte *p, int32 n)
{
	int32 i;

	for(i=0; i<n; i++) {
		runtime·printpointer_c((byte*)(p[i]>>4));
		runtime·printpointer_c((byte*)(p[i]&0xf));
		if((i&15) == 15)
			runtime·prints("\n");
		else
			runtime·prints(" ");
	}
	if(n & 15)
		runtime·prints("\n");
}

void
runtime·prints(int8 *s)
{
	gwrite(s, runtime·findnull((byte*)s));
}

#pragma textflag NOSPLIT
void
runtime·printf(int8 *s, ...)
{
	byte *arg;

	arg = (byte*)(&s+1);
	vprintf(s, arg);
}

#pragma textflag NOSPLIT
int32
runtime·snprintf(byte *buf, int32 n, int8 *s, ...)
{
	byte *arg;
	int32 m;

	arg = (byte*)(&s+1);
	g->writebuf = buf;
	g->writenbuf = n-1;
	vprintf(s, arg);
	*g->writebuf = '\0';
	m = g->writebuf - buf;
	g->writenbuf = 0;
	g->writebuf = nil;
	return m;
}

// Very simple printf.  Only for debugging prints.
// Do not add to this without checking with Rob.
static void
vprintf(int8 *s, byte *base)
{
	int8 *p, *lp;
	uintptr arg, siz;
	byte *v;

	//runtime·lock(&debuglock);

	lp = p = s;
	arg = (uintptr)base;
	for(; *p; p++) {
		if(*p != '%')
			continue;
		if(p > lp)
			gwrite(lp, p-lp);
		p++;
		siz = 0;
		switch(*p) {
		case 't':
		case 'c':
			siz = 1;
			break;
		case 'd':	// 32-bit
		case 'x':
			arg = ROUND(arg, 4);
			siz = 4;
			break;
		case 'D':	// 64-bit
		case 'U':
		case 'X':
		case 'f':
			arg = ROUND(arg, sizeof(uintreg));
			siz = 8;
			break;
		case 'C':
			arg = ROUND(arg, sizeof(uintreg));
			siz = 16;
			break;
		case 'p':	// pointer-sized
		case 's':
			arg = ROUND(arg, sizeof(uintptr));
			siz = sizeof(uintptr);
			break;
		case 'S':	// pointer-aligned but bigger
			arg = ROUND(arg, sizeof(uintptr));
			siz = sizeof(String);
			break;
		case 'a':	// pointer-aligned but bigger
			arg = ROUND(arg, sizeof(uintptr));
			siz = sizeof(Slice);
			break;
		case 'i':	// pointer-aligned but bigger
		case 'e':
			arg = ROUND(arg, sizeof(uintptr));
			siz = sizeof(Eface);
			break;
		}
		v = (byte*)arg;
		switch(*p) {
		case 'a':
			runtime·printslice_c(*(Slice*)v);
			break;
		case 'c':
			runtime·printbyte_c(*(int8*)v);
			break;
		case 'd':
			runtime·printint_c(*(int32*)v);
			break;
		case 'D':
			runtime·printint_c(*(int64*)v);
			break;
		case 'e':
			runtime·printeface_c(*(Eface*)v);
			break;
		case 'f':
			runtime·printfloat_c(*(float64*)v);
			break;
		case 'C':
			runtime·printcomplex_c(*(Complex128*)v);
			break;
		case 'i':
			runtime·printiface_c(*(Iface*)v);
			break;
		case 'p':
			runtime·printpointer_c(*(void**)v);
			break;
		case 's':
			runtime·prints(*(int8**)v);
			break;
		case 'S':
			runtime·printstring_c(*(String*)v);
			break;
		case 't':
			runtime·printbool_c(*(bool*)v);
			break;
		case 'U':
			runtime·printuint_c(*(uint64*)v);
			break;
		case 'x':
			runtime·printhex_c(*(uint32*)v);
			break;
		case 'X':
			runtime·printhex_c(*(uint64*)v);
			break;
		}
		arg += siz;
		lp = p+1;
	}
	if(p > lp)
		gwrite(lp, p-lp);

	//runtime·unlock(&debuglock);
}

static void
goprintf_m(void)
{
	// Can assume s has terminating NUL because only
	// the Go compiler generates calls to runtime·goprintf, using
	// string constants, and all the string constants have NULs.
	vprintf(g->m->ptrarg[0], g->m->ptrarg[1]);
	g->m->ptrarg[0] = nil;
	g->m->ptrarg[1] = nil;
}

#pragma textflag NOSPLIT
void
runtime·goprintf(String s, ...)
{
	g->m->ptrarg[0] = s.str;
	g->m->ptrarg[1] = (byte*)(&s+1);
	runtime·onM(goprintf_m);
}

void
runtime·printpc_c(void *p)
{
	runtime·prints("PC=");
	runtime·printhex_c((uint64)runtime·getcallerpc(p));
}

void
runtime·printbool_c(bool v)
{
	if(v) {
		gwrite((byte*)"true", 4);
		return;
	}
	gwrite((byte*)"false", 5);
}

void
runtime·printbyte_c(int8 c)
{
	gwrite(&c, 1);
}

void
runtime·printfloat_c(float64 v)
{
	byte buf[20];
	int32 e, s, i, n;
	float64 h;

	if(ISNAN(v)) {
		gwrite("NaN", 3);
		return;
	}
	if(v == runtime·posinf) {
		gwrite("+Inf", 4);
		return;
	}
	if(v == runtime·neginf) {
		gwrite("-Inf", 4);
		return;
	}

	n = 7;	// digits printed
	e = 0;	// exp
	s = 0;	// sign
	if(v == 0) {
		if(1/v == runtime·neginf)
			s = 1;
	} else {
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
	gwrite(buf, n+7);
}

void
runtime·printcomplex_c(Complex128 v)
{
	gwrite("(", 1);
	runtime·printfloat_c(v.real);
	runtime·printfloat_c(v.imag);
	gwrite("i)", 2);
}

void
runtime·printuint_c(uint64 v)
{
	byte buf[100];
	int32 i;

	for(i=nelem(buf)-1; i>0; i--) {
		buf[i] = v%10 + '0';
		if(v < 10)
			break;
		v = v/10;
	}
	gwrite(buf+i, nelem(buf)-i);
}

void
runtime·printint_c(int64 v)
{
	if(v < 0) {
		gwrite("-", 1);
		v = -v;
	}
	runtime·printuint_c(v);
}

void
runtime·printhex_c(uint64 v)
{
	static int8 *dig = "0123456789abcdef";
	byte buf[100];
	int32 i;

	i=nelem(buf);
	for(; v>0; v/=16)
		buf[--i] = dig[v%16];
	if(i == nelem(buf))
		buf[--i] = '0';
	buf[--i] = 'x';
	buf[--i] = '0';
	gwrite(buf+i, nelem(buf)-i);
}

void
runtime·printpointer_c(void *p)
{
	runtime·printhex_c((uintptr)p);
}

void
runtime·printstring_c(String v)
{
	if(v.len > runtime·maxstring) {
		gwrite("[string too long]", 17);
		return;
	}
	if(v.len > 0)
		gwrite(v.str, v.len);
}

void
runtime·printslice_c(Slice s)
{
	runtime·prints("[");
	runtime·printint_c(s.len);
	runtime·prints("/");
	runtime·printint_c(s.cap);
	runtime·prints("]");
	runtime·printpointer_c(s.array);
}

void
runtime·printeface_c(Eface e)
{
	runtime·printf("(%p,%p)", e.type, e.data);
}

void
runtime·printiface_c(Iface i)
{
	runtime·printf("(%p,%p)", i.tab, i.data);
}

void
runtime·printstring_m(void)
{
	String s;

	s.str = g->m->ptrarg[0];
	g->m->ptrarg[0] = nil;
	s.len = g->m->scalararg[0];
	runtime·printstring_c(s);
}

void
runtime·printuint_m(void)
{
	runtime·printuint_c(*(uint64*)(&g->m->scalararg[0]));
}

void
runtime·printhex_m(void)
{
	runtime·printhex_c(g->m->scalararg[0]);
}

void
runtime·printfloat_m(void)
{
	runtime·printfloat_c(*(float64*)(&g->m->scalararg[0]));
}
