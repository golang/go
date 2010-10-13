// Inferno utils/5c/mul.c
// http://code.google.com/p/inferno-os/source/browse/utils/5c/mul.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.


#include "gc.h"

/*
 * code sequences for multiply by constant.
 * [a-l][0-3]
 *	lsl	$(A-'a'),r0,r1
 * [+][0-7]
 *	add	r0,r1,r2
 * [-][0-7]
 *	sub	r0,r1,r2
 */

static  int	maxmulops = 3;	/* max # of ops to replace mul with */
static	int	multabp;
static	int32	mulval;
static	char*	mulcp;
static	int32	valmax;
static	int	shmax;

static int	docode(char *hp, char *cp, int r0, int r1);
static int	gen1(int len);
static int	gen2(int len, int32 r1);
static int	gen3(int len, int32 r0, int32 r1, int flag);
enum
{
	SR1	= 1<<0,		/* r1 has been shifted */
	SR0	= 1<<1,		/* r0 has been shifted */
	UR1	= 1<<2,		/* r1 has not been used */
	UR0	= 1<<3,		/* r0 has not been used */
};

Multab*
mulcon0(int32 v)
{
	int a1, a2, g;
	Multab *m, *m1;
	char hint[10];

	if(v < 0)
		v = -v;

	/*
	 * look in cache
	 */
	m = multab;
	for(g=0; g<nelem(multab); g++) {
		if(m->val == v) {
			if(m->code[0] == 0)
				return 0;
			return m;
		}
		m++;
	}

	/*
	 * select a spot in cache to overwrite
	 */
	multabp++;
	if(multabp < 0 || multabp >= nelem(multab))
		multabp = 0;
	m = multab+multabp;
	m->val = v;
	mulval = v;

	/*
	 * look in execption hint table
	 */
	a1 = 0;
	a2 = hintabsize;
	for(;;) {
		if(a1 >= a2)
			goto no;
		g = (a2 + a1)/2;
		if(v < hintab[g].val) {
			a2 = g;
			continue;
		}
		if(v > hintab[g].val) {
			a1 = g+1;
			continue;
		}
		break;
	}

	if(docode(hintab[g].hint, m->code, 1, 0))
		return m;
	print("multiply table failure %d\n", v);
	m->code[0] = 0;
	return 0;

no:
	/*
	 * try to search
	 */
	hint[0] = 0;
	for(g=1; g<=maxmulops; g++) {
		if(g >= maxmulops && v >= 65535)
			break;
		mulcp = hint+g;
		*mulcp = 0;
		if(gen1(g)) {
			if(docode(hint, m->code, 1, 0))
				return m;
			print("multiply table failure %d\n", v);
			break;
		}
	}

	/*
	 * try a recur followed by a shift
	 */
	g = 0;
	while(!(v & 1)) {
		g++;
		v >>= 1;
	}
	if(g) {
		m1 = mulcon0(v);
		if(m1) {
			strcpy(m->code, m1->code);
			sprint(strchr(m->code, 0), "%c0", g+'a');
			return m;
		}
	}
	m->code[0] = 0;
	return 0;
}

static int
docode(char *hp, char *cp, int r0, int r1)
{
	int c, i;

	c = *hp++;
	*cp = c;
	cp += 2;
	switch(c) {
	default:
		c -= 'a';
		if(c < 1 || c >= 30)
			break;
		for(i=0; i<4; i++) {
			switch(i) {
			case 0:
				if(docode(hp, cp, r0<<c, r1))
					goto out;
				break;
			case 1:
				if(docode(hp, cp, r1<<c, r1))
					goto out;
				break;
			case 2:
				if(docode(hp, cp, r0, r0<<c))
					goto out;
				break;
			case 3:
				if(docode(hp, cp, r0, r1<<c))
					goto out;
				break;
			}
		}
		break;

	case '+':
		for(i=0; i<8; i++) {
			cp[-1] = i+'0';
			switch(i) {
			case 1:
				if(docode(hp, cp, r0+r1, r1))
					goto out;
				break;
			case 5:
				if(docode(hp, cp, r0, r0+r1))
					goto out;
				break;
			}
		}
		break;

	case '-':
		for(i=0; i<8; i++) {
			cp[-1] = i+'0';
			switch(i) {
			case 1:
				if(docode(hp, cp, r0-r1, r1))
					goto out;
				break;
			case 2:
				if(docode(hp, cp, r1-r0, r1))
					goto out;
				break;
			case 5:
				if(docode(hp, cp, r0, r0-r1))
					goto out;
				break;
			case 6:
				if(docode(hp, cp, r0, r1-r0))
					goto out;
				break;
			}
		}
		break;

	case 0:
		if(r0 == mulval)
			return 1;
	}
	return 0;

out:
	cp[-1] = i+'0';
	return 1;
}

static int
gen1(int len)
{
	int i;

	for(shmax=1; shmax<30; shmax++) {
		valmax = 1<<shmax;
		if(valmax >= mulval)
			break;
	}
	if(mulval == 1)
		return 1;

	len--;
	for(i=1; i<=shmax; i++)
		if(gen2(len, 1<<i)) {
			*--mulcp = 'a'+i;
			return 1;
		}
	return 0;
}

static int
gen2(int len, int32 r1)
{
	int i;

	if(len <= 0) {
		if(r1 == mulval)
			return 1;
		return 0;
	}

	len--;
	if(len == 0)
		goto calcr0;

	if(gen3(len, r1, r1+1, UR1)) {
		i = '+';
		goto out;
	}
	if(gen3(len, r1-1, r1, UR0)) {
		i = '-';
		goto out;
	}
	if(gen3(len, 1, r1+1, UR1)) {
		i = '+';
		goto out;
	}
	if(gen3(len, 1, r1-1, UR1)) {
		i = '-';
		goto out;
	}

	return 0;

calcr0:
	if(mulval == r1+1) {
		i = '+';
		goto out;
	}
	if(mulval == r1-1) {
		i = '-';
		goto out;
	}
	return 0;

out:
	*--mulcp = i;
	return 1;
}

static int
gen3(int len, int32 r0, int32 r1, int flag)
{
	int i, f1, f2;
	int32 x;

	if(r0 <= 0 ||
	   r0 >= r1 ||
	   r1 > valmax)
		return 0;

	len--;
	if(len == 0)
		goto calcr0;

	if(!(flag & UR1)) {
		f1 = UR1|SR1;
		for(i=1; i<=shmax; i++) {
			x = r0<<i;
			if(x > valmax)
				break;
			if(gen3(len, r0, x, f1)) {
				i += 'a';
				goto out;
			}
		}
	}

	if(!(flag & UR0)) {
		f1 = UR1|SR1;
		for(i=1; i<=shmax; i++) {
			x = r1<<i;
			if(x > valmax)
				break;
			if(gen3(len, r1, x, f1)) {
				i += 'a';
				goto out;
			}
		}
	}

	if(!(flag & SR1)) {
		f1 = UR1|SR1|(flag&UR0);
		for(i=1; i<=shmax; i++) {
			x = r1<<i;
			if(x > valmax)
				break;
			if(gen3(len, r0, x, f1)) {
				i += 'a';
				goto out;
			}
		}
	}

	if(!(flag & SR0)) {
		f1 = UR0|SR0|(flag&(SR1|UR1));

		f2 = UR1|SR1;
		if(flag & UR1)
			f2 |= UR0;
		if(flag & SR1)
			f2 |= SR0;

		for(i=1; i<=shmax; i++) {
			x = r0<<i;
			if(x > valmax)
				break;
			if(x > r1) {
				if(gen3(len, r1, x, f2)) {
					i += 'a';
					goto out;
				}
			} else
				if(gen3(len, x, r1, f1)) {
					i += 'a';
					goto out;
				}
		}
	}

	x = r1+r0;
	if(gen3(len, r0, x, UR1)) {
		i = '+';
		goto out;
	}

	if(gen3(len, r1, x, UR1)) {
		i = '+';
		goto out;
	}

	x = r1-r0;
	if(gen3(len, x, r1, UR0)) {
		i = '-';
		goto out;
	}

	if(x > r0) {
		if(gen3(len, r0, x, UR1)) {
			i = '-';
			goto out;
		}
	} else
		if(gen3(len, x, r0, UR0)) {
			i = '-';
			goto out;
		}

	return 0;

calcr0:
	f1 = flag & (UR0|UR1);
	if(f1 == UR1) {
		for(i=1; i<=shmax; i++) {
			x = r1<<i;
			if(x >= mulval) {
				if(x == mulval) {
					i += 'a';
					goto out;
				}
				break;
			}
		}
	}

	if(mulval == r1+r0) {
		i = '+';
		goto out;
	}
	if(mulval == r1-r0) {
		i = '-';
		goto out;
	}

	return 0;

out:
	*--mulcp = i;
	return 1;
}

/*
 * hint table has numbers that
 * the search algorithm fails on.
 * <1000:
 *	all numbers
 * <5000:
 * 	÷ by 5
 * <10000:
 * 	÷ by 50
 * <65536:
 * 	÷ by 250
 */
Hintab	hintab[] =
{
	683,	"b++d+e+",
	687,	"b+e++e-",
	691,	"b++d+e+",
	731,	"b++d+e+",
	811,	"b++d+i+",
	821,	"b++e+e+",
	843,	"b+d++e+",
	851,	"b+f-+e-",
	853,	"b++e+e+",
	877,	"c++++g-",
	933,	"b+c++g-",
	981,	"c-+e-d+",
	1375,	"b+c+b+h-",
	1675,	"d+b++h+",
	2425,	"c++f-e+",
	2675,	"c+d++f-",
	2750,	"b+d-b+h-",
	2775,	"c-+g-e-",
	3125,	"b++e+g+",
	3275,	"b+c+g+e+",
	3350,	"c++++i+",
	3475,	"c-+e-f-",
	3525,	"c-+d+g-",
	3625,	"c-+e-j+",
	3675,	"b+d+d+e+",
	3725,	"b+d-+h+",
	3925,	"b+d+f-d-",
	4275,	"b+g++e+",
	4325,	"b+h-+d+",
	4425,	"b+b+g-j-",
	4525,	"b+d-d+f+",
	4675,	"c++d-g+",
	4775,	"b+d+b+g-",
	4825,	"c+c-+i-",
	4850,	"c++++i-",
	4925,	"b++e-g-",
	4975,	"c+f++e-",
	5500,	"b+g-c+d+",
	6700,	"d+b++i+",
	9700,	"d++++j-",
	11000,	"b+f-c-h-",
	11750,	"b+d+g+j-",
	12500,	"b+c+e-k+",
	13250,	"b+d+e-f+",
	13750,	"b+h-c-d+",
	14250,	"b+g-c+e-",
	14500,	"c+f+j-d-",
	14750,	"d-g--f+",
	16750,	"b+e-d-n+",
	17750,	"c+h-b+e+",
	18250,	"d+b+h-d+",
	18750,	"b+g-++f+",
	19250,	"b+e+b+h+",
	19750,	"b++h--f-",
	20250,	"b+e-l-c+",
	20750,	"c++bi+e-",
	21250,	"b+i+l+c+",
	22000,	"b+e+d-g-",
	22250,	"b+d-h+k-",
	22750,	"b+d-e-g+",
	23250,	"b+c+h+e-",
	23500,	"b+g-c-g-",
	23750,	"b+g-b+h-",
	24250,	"c++g+m-",
	24750,	"b+e+e+j-",
	25000,	"b++dh+g+",
	25250,	"b+e+d-g-",
	25750,	"b+e+b+j+",
	26250,	"b+h+c+e+",
	26500,	"b+h+c+g+",
	26750,	"b+d+e+g-",
	27250,	"b+e+e+f+",
	27500,	"c-i-c-d+",
	27750,	"b+bd++j+",
	28250,	"d-d-++i-",
	28500,	"c+c-h-e-",
	29000,	"b+g-d-f+",
	29500,	"c+h+++e-",
	29750,	"b+g+f-c+",
	30250,	"b+f-g-c+",
	33500,	"c-f-d-n+",
	33750,	"b+d-b+j-",
	34250,	"c+e+++i+",
	35250,	"e+b+d+k+",
	35500,	"c+e+d-g-",
	35750,	"c+i-++e+",
	36250,	"b+bh-d+e+",
	36500,	"c+c-h-e-",
	36750,	"d+e--i+",
	37250,	"b+g+g+b+",
	37500,	"b+h-b+f+",
	37750,	"c+be++j-",
	38500,	"b+e+b+i+",
	38750,	"d+i-b+d+",
	39250,	"b+g-l-+d+",
	39500,	"b+g-c+g-",
	39750,	"b+bh-c+f-",
	40250,	"b+bf+d+g-",
	40500,	"b+g-c+g+",
	40750,	"c+b+i-e+",
	41250,	"d++bf+h+",
	41500,	"b+j+c+d-",
	41750,	"c+f+b+h-",
	42500,	"c+h++g+",
	42750,	"b+g+d-f-",
	43250,	"b+l-e+d-",
	43750,	"c+bd+h+f-",
	44000,	"b+f+g-d-",
	44250,	"b+d-g--f+",
	44500,	"c+e+c+h+",
	44750,	"b+e+d-h-",
	45250,	"b++g+j-g+",
	45500,	"c+d+e-g+",
	45750,	"b+d-h-e-",
	46250,	"c+bd++j+",
	46500,	"b+d-c-j-",
	46750,	"e-e-b+g-",
	47000,	"b+c+d-j-",
	47250,	"b+e+e-g-",
	47500,	"b+g-c-h-",
	47750,	"b+f-c+h-",
	48250,	"d--h+n-",
	48500,	"b+c-g+m-",
	48750,	"b+e+e-g+",
	49500,	"c-f+e+j-",
	49750,	"c+c+g++f-",
	50000,	"b+e+e+k+",
	50250,	"b++i++g+",
	50500,	"c+g+f-i+",
	50750,	"b+e+d+k-",
	51500,	"b+i+c-f+",
	51750,	"b+bd+g-e-",
	52250,	"b+d+g-j+",
	52500,	"c+c+f+g+",
	52750,	"b+c+e+i+",
	53000,	"b+i+c+g+",
	53500,	"c+g+g-n+",
	53750,	"b+j+d-c+",
	54250,	"b+d-g-j-",
	54500,	"c-f+e+f+",
	54750,	"b+f-+c+g+",
	55000,	"b+g-d-g-",
	55250,	"b+e+e+g+",
	55500,	"b+cd++j+",
	55750,	"b+bh-d-f-",
	56250,	"c+d-b+j-",
	56500,	"c+d+c+i+",
	56750,	"b+e+d++h-",
	57000,	"b+d+g-f+",
	57250,	"b+f-m+d-",
	57750,	"b+i+c+e-",
	58000,	"b+e+d+h+",
	58250,	"c+b+g+g+",
	58750,	"d-e-j--e+",
	59000,	"d-i-+e+",
	59250,	"e--h-m+",
	59500,	"c+c-h+f-",
	59750,	"b+bh-e+i-",
	60250,	"b+bh-e-e-",
	60500,	"c+c-g-g-",
	60750,	"b+e-l-e-",
	61250,	"b+g-g-c+",
	61750,	"b+g-c+g+",
	62250,	"f--+c-i-",
	62750,	"e+f--+g+",
	64750,	"b+f+d+p-",
};
int	hintabsize	= nelem(hintab);
