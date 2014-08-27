// Inferno utils/5c/sgen.c
// http://code.google.com/p/inferno-os/source/browse/utils/5c/sgen.c
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

Prog*
gtext(Sym *s, int32 stkoff)
{
	int32 a;

	a = argsize(1);
	if((textflag & NOSPLIT) != 0 && stkoff >= 128)
		yyerror("stack frame too large for NOSPLIT function");

	gpseudo(ATEXT, s, nodconst(stkoff));
	p->to.type = D_CONST2;
	p->to.offset2 = a;
	return p;
}

void
noretval(int n)
{

	if(n & 1) {
		gins(ANOP, Z, Z);
		p->to.type = D_REG;
		p->to.reg = REGRET;
	}
	if(n & 2) {
		gins(ANOP, Z, Z);
		p->to.type = D_FREG;
		p->to.reg = FREGRET;
	}
}

/*
 *	calculate addressability as follows
 *		CONST ==> 20		$value
 *		NAME ==> 10		name
 *		REGISTER ==> 11		register
 *		INDREG ==> 12		*[(reg)+offset]
 *		&10 ==> 2		$name
 *		ADD(2, 20) ==> 2	$name+offset
 *		ADD(3, 20) ==> 3	$(reg)+offset
 *		&12 ==> 3		$(reg)+offset
 *		*11 ==> 11		??
 *		*2 ==> 10		name
 *		*3 ==> 12		*(reg)+offset
 *	calculate complexity (number of registers)
 */
void
xcom(Node *n)
{
	Node *l, *r;
	int t;

	if(n == Z)
		return;
	l = n->left;
	r = n->right;
	n->addable = 0;
	n->complex = 0;
	switch(n->op) {
	case OCONST:
		n->addable = 20;
		return;

	case OREGISTER:
		n->addable = 11;
		return;

	case OINDREG:
		n->addable = 12;
		return;

	case ONAME:
		n->addable = 10;
		return;

	case OADDR:
		xcom(l);
		if(l->addable == 10)
			n->addable = 2;
		if(l->addable == 12)
			n->addable = 3;
		break;

	case OIND:
		xcom(l);
		if(l->addable == 11)
			n->addable = 12;
		if(l->addable == 3)
			n->addable = 12;
		if(l->addable == 2)
			n->addable = 10;
		break;

	case OADD:
		xcom(l);
		xcom(r);
		if(l->addable == 20) {
			if(r->addable == 2)
				n->addable = 2;
			if(r->addable == 3)
				n->addable = 3;
		}
		if(r->addable == 20) {
			if(l->addable == 2)
				n->addable = 2;
			if(l->addable == 3)
				n->addable = 3;
		}
		break;

	case OASLMUL:
	case OASMUL:
		xcom(l);
		xcom(r);
		t = vlog(r);
		if(t >= 0) {
			n->op = OASASHL;
			r->vconst = t;
			r->type = types[TINT];
		}
		break;

	case OMUL:
	case OLMUL:
		xcom(l);
		xcom(r);
		t = vlog(r);
		if(t >= 0) {
			n->op = OASHL;
			r->vconst = t;
			r->type = types[TINT];
		}
		t = vlog(l);
		if(t >= 0) {
			n->op = OASHL;
			n->left = r;
			n->right = l;
			r = l;
			l = n->left;
			r->vconst = t;
			r->type = types[TINT];
		}
		break;

	case OASLDIV:
		xcom(l);
		xcom(r);
		t = vlog(r);
		if(t >= 0) {
			n->op = OASLSHR;
			r->vconst = t;
			r->type = types[TINT];
		}
		break;

	case OLDIV:
		xcom(l);
		xcom(r);
		t = vlog(r);
		if(t >= 0) {
			n->op = OLSHR;
			r->vconst = t;
			r->type = types[TINT];
		}
		break;

	case OASLMOD:
		xcom(l);
		xcom(r);
		t = vlog(r);
		if(t >= 0) {
			n->op = OASAND;
			r->vconst--;
		}
		break;

	case OLMOD:
		xcom(l);
		xcom(r);
		t = vlog(r);
		if(t >= 0) {
			n->op = OAND;
			r->vconst--;
		}
		break;

	default:
		if(l != Z)
			xcom(l);
		if(r != Z)
			xcom(r);
		break;
	}
	if(n->addable >= 10)
		return;

	if(l != Z)
		n->complex = l->complex;
	if(r != Z) {
		if(r->complex == n->complex)
			n->complex = r->complex+1;
		else
		if(r->complex > n->complex)
			n->complex = r->complex;
	}
	if(n->complex == 0)
		n->complex++;

	if(com64(n))
		return;

	switch(n->op) {
	case OFUNC:
		n->complex = FNX;
		break;

	case OADD:
	case OXOR:
	case OAND:
	case OOR:
	case OEQ:
	case ONE:
		/*
		 * immediate operators, make const on right
		 */
		if(l->op == OCONST) {
			n->left = r;
			n->right = l;
		}
		break;
	}
}
