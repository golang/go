// Inferno utils/cc/bits.c
// http://code.google.com/p/inferno-os/source/browse/utils/cc/bits.c
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

package gc

import "fmt"

/*
Bits
bor(Bits a, Bits b)
{
	Bits c;
	int i;

	for(i=0; i<BITS; i++)
		c.b[i] = a.b[i] | b.b[i];
	return c;
}

Bits
band(Bits a, Bits b)
{
	Bits c;
	int i;

	for(i=0; i<BITS; i++)
		c.b[i] = a.b[i] & b.b[i];
	return c;
}

Bits
bnot(Bits a)
{
	Bits c;
	int i;

	for(i=0; i<BITS; i++)
		c.b[i] = ~a.b[i];
	return c;
}
*/
func bany(a *Bits) bool {
	for i := 0; i < BITS; i++ {
		if a.b[i] != 0 {
			return true
		}
	}
	return false
}

/*
int
beq(Bits a, Bits b)
{
	int i;

	for(i=0; i<BITS; i++)
		if(a.b[i] != b.b[i])
			return 0;
	return 1;
}
*/
func bnum(a Bits) int {
	var b uint64

	for i := 0; i < BITS; i++ {
		b = a.b[i]
		if b != 0 {
			return 64*i + Bitno(b)
		}
	}

	Fatal("bad in bnum")
	return 0
}

func blsh(n uint) Bits {
	c := zbits
	c.b[n/64] = 1 << (n % 64)
	return c
}

func btest(a *Bits, n uint) bool {
	return a.b[n/64]&(1<<(n%64)) != 0
}

func biset(a *Bits, n uint) {
	a.b[n/64] |= 1 << (n % 64)
}

func biclr(a *Bits, n uint) {
	a.b[n/64] &^= (1 << (n % 64))
}

func Bitno(b uint64) int {
	for i := 0; i < 64; i++ {
		if b&(1<<uint(i)) != 0 {
			return i
		}
	}
	Fatal("bad in bitno")
	return 0
}

func Qconv(bits Bits, flag int) string {
	var fp string

	var i int

	first := 1

	for bany(&bits) {
		i = bnum(bits)
		if first != 0 {
			first = 0
		} else {
			fp += fmt.Sprintf(" ")
		}
		if var_[i].node == nil || var_[i].node.Sym == nil {
			fp += fmt.Sprintf("$%d", i)
		} else {
			fp += fmt.Sprintf("%s(%d)", var_[i].node.Sym.Name, i)
			if var_[i].offset != 0 {
				fp += fmt.Sprintf("%+d", int64(var_[i].offset))
			}
		}

		biclr(&bits, uint(i))
	}

	return fp
}
