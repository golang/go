// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go.h"

enum
{
	PFIXME = 0,
};

void
exprlistfmt(Fmt *f, NodeList *l)
{
	for(; l; l=l->next) {
		exprfmt(f, l->n, 0);
		if(l->next)
			fmtprint(f, ", ");
	}
}

void
exprfmt(Fmt *f, Node *n, int prec)
{
	int nprec;
	char *p;

	nprec = 0;
	if(n == nil) {
		fmtprint(f, "<nil>");
		return;
	}
	
	if(n->implicit) {
		exprfmt(f, n->left, prec);
		return;
	}

	switch(n->op) {
	case ONAME:
	case ONONAME:
	case OPACK:
	case OLITERAL:
	case ODOT:
	case ODOTPTR:
	case ODOTINTER:
	case ODOTMETH:
	case ODOTTYPE:
	case ODOTTYPE2:
	case OARRAYBYTESTR:
	case OCAP:
	case OCLOSE:
	case OCLOSED:
	case OCOPY:
	case OLEN:
	case OMAKE:
	case ONEW:
	case OPANIC:
	case OPRINT:
	case OPRINTN:
	case OCALL:
	case OCALLMETH:
	case OCALLINTER:
	case OCALLFUNC:
	case OCONV:
	case OCONVNOP:
	case OMAKESLICE:
	case ORUNESTR:
	case OADDR:
	case OCOM:
	case OIND:
	case OMINUS:
	case ONOT:
	case OPLUS:
	case ORECV:
	case OCONVIFACE:
	case OTPAREN:
	case OINDEX:
	case OINDEXMAP:
		nprec = 7;
		break;

	case OMUL:
	case ODIV:
	case OMOD:
	case OLSH:
	case ORSH:
	case OAND:
	case OANDNOT:
		nprec = 6;
		break;

	case OADD:
	case OSUB:
	case OOR:
	case OXOR:
		nprec = 5;
		break;

	case OEQ:
	case OLT:
	case OLE:
	case OGE:
	case OGT:
	case ONE:
		nprec = 4;
		break;

	case OSEND:
		nprec = 3;
		break;

	case OANDAND:
		nprec = 2;
		break;

	case OOROR:
		nprec = 1;
		break;
	
	case OTYPE:
		if(n->sym != S)
			nprec = 7;
		break;
	}

	if(prec > nprec)
		fmtprint(f, "(");

	switch(n->op) {
	default:
	bad:
		fmtprint(f, "(node %O)", n->op);
		break;

	case OLITERAL:
		if(n->sym != S) {
			fmtprint(f, "%S", n->sym);
			break;
		}
		switch(n->val.ctype) {
		default:
			goto bad;
		case CTINT:
			fmtprint(f, "%B", n->val.u.xval);
			break;
		case CTBOOL:
			if(n->val.u.bval)
				fmtprint(f, "true");
			else
				fmtprint(f, "false");
			break;
		case CTCPLX:
			fmtprint(f, "%.17g+%.17gi",
				mpgetflt(&n->val.u.cval->real),
				mpgetflt(&n->val.u.cval->imag));
			break;
		case CTFLT:
			fmtprint(f, "%.17g", mpgetflt(n->val.u.fval));
			break;
		case CTSTR:
			fmtprint(f, "\"%Z\"", n->val.u.sval);
			break;
		case CTNIL:
			fmtprint(f, "nil");
			break;
		}
		break;

	case ONAME:
	case OPACK:
	case ONONAME:
		fmtprint(f, "%S", n->sym);
		break;

	case OTYPE:
		if(n->type == T && n->sym != S) {
			fmtprint(f, "%S", n->sym);
			break;
		}
		fmtprint(f, "%T", n->type);
		break;

	case OTARRAY:
		fmtprint(f, "[]");
		exprfmt(f, n->left, PFIXME);
		break;
	
	case OTPAREN:
		fmtprint(f, "(");
		exprfmt(f, n->left, 0);
		fmtprint(f, ")");
		break;

	case OTMAP:
		fmtprint(f, "map[");
		exprfmt(f, n->left, 0);
		fmtprint(f, "] ");
		exprfmt(f, n->right, 0);
		break;

	case OTCHAN:
		if(n->etype == Crecv)
			fmtprint(f, "<-");
		fmtprint(f, "chan");
		if(n->etype == Csend) {
			fmtprint(f, "<- ");
			exprfmt(f, n->left, 0);
		} else {
			fmtprint(f, " ");
			if(n->left->op == OTCHAN && n->left->sym == S && n->left->etype == Crecv) {
				fmtprint(f, "(");
				exprfmt(f, n->left, 0);
				fmtprint(f, ")");
			} else
				exprfmt(f, n->left, 0);
		}
		break;

	case OTSTRUCT:
		fmtprint(f, "<struct>");
		break;

	case OTINTER:
		fmtprint(f, "<inter>");
		break;

	case OTFUNC:
		fmtprint(f, "<func>");
		break;

	case OAS:
		exprfmt(f, n->left, 0);
		fmtprint(f, " = ");
		exprfmt(f, n->right, 0);
		break;

	case OASOP:
		exprfmt(f, n->left, 0);
		fmtprint(f, " %#O= ", n->etype);
		exprfmt(f, n->right, 0);
		break;

	case OADD:
	case OANDAND:
	case OANDNOT:
	case ODIV:
	case OEQ:
	case OGE:
	case OGT:
	case OLE:
	case OLT:
	case OLSH:
	case OMOD:
	case OMUL:
	case ONE:
	case OOR:
	case OOROR:
	case ORSH:
	case OSEND:
	case OSUB:
	case OXOR:
		exprfmt(f, n->left, nprec);
		fmtprint(f, " %#O ", n->op);
		exprfmt(f, n->right, nprec+1);
		break;

	case OADDR:
	case OCOM:
	case OIND:
	case OMINUS:
	case ONOT:
	case OPLUS:
	case ORECV:
		fmtprint(f, "%#O", n->op);
		if((n->op == OMINUS || n->op == OPLUS) && n->left->op == n->op)
			fmtprint(f, " ");
		exprfmt(f, n->left, 0);
		break;

	case OCLOSURE:
		fmtprint(f, "func literal");
		break;

	case OCOMPLIT:
		fmtprint(f, "composite literal");
		break;
	
	case OARRAYLIT:
		if(isslice(n->type))
			fmtprint(f, "slice literal");
		else
			fmtprint(f, "array literal");
		break;
	
	case OMAPLIT:
		fmtprint(f, "map literal");
		break;
	
	case OSTRUCTLIT:
		fmtprint(f, "struct literal");
		break;

	case OXDOT:
	case ODOT:
	case ODOTPTR:
	case ODOTINTER:
	case ODOTMETH:
		exprfmt(f, n->left, 7);
		if(n->right == N || n->right->sym == S)
			fmtprint(f, ".<nil>");
		else {
			// skip leading type· in method name
			p = utfrrune(n->right->sym->name, 0xb7);
			if(p)
				p+=2;
			else
				p = n->right->sym->name;
			fmtprint(f, ".%s", p);
		}
		break;

	case ODOTTYPE:
	case ODOTTYPE2:
		exprfmt(f, n->left, 7);
		fmtprint(f, ".(");
		if(n->right != N)
			exprfmt(f, n->right, 0);
		else
			fmtprint(f, "%T", n->type);
		fmtprint(f, ")");
		break;

	case OINDEX:
	case OINDEXMAP:
		exprfmt(f, n->left, 7);
		fmtprint(f, "[");
		exprfmt(f, n->right, 0);
		fmtprint(f, "]");
		break;

	case OSLICE:
	case OSLICESTR:
	case OSLICEARR:
		exprfmt(f, n->left, 7);
		fmtprint(f, "[");
		if(n->right->left != N)
			exprfmt(f, n->right->left, 0);
		fmtprint(f, ":");
		if(n->right->right != N)
			exprfmt(f, n->right->right, 0);
		fmtprint(f, "]");
		break;

	case OCALL:
	case OCALLFUNC:
	case OCALLINTER:
	case OCALLMETH:
		exprfmt(f, n->left, 7);
		fmtprint(f, "(");
		exprlistfmt(f, n->list);
		if(n->isddd)
			fmtprint(f, "...");
		fmtprint(f, ")");
		break;

	case OCMPLX:
		fmtprint(f, "cmplx(");
		exprfmt(f, n->left, 0);
		fmtprint(f, ", ");
		exprfmt(f, n->right, 0);
		fmtprint(f, ")");
		break;

	case OREAL:
		fmtprint(f, "real(");
		exprfmt(f, n->left, 0);
		fmtprint(f, ")");
		break;

	case OIMAG:
		fmtprint(f, "imag(");
		exprfmt(f, n->left, 0);
		fmtprint(f, ")");
		break;

	case OCONV:
	case OCONVIFACE:
	case OCONVNOP:
	case OARRAYBYTESTR:
	case ORUNESTR:
		if(n->type == T || n->type->sym == S)
			fmtprint(f, "(%T)(", n->type);
		else
			fmtprint(f, "%T(", n->type);
		if(n->left == N)
			exprlistfmt(f, n->list);
		else
			exprfmt(f, n->left, 0);
		fmtprint(f, ")");
		break;

	case OCAP:
	case OCLOSE:
	case OCLOSED:
	case OLEN:
	case OCOPY:
	case OMAKE:
	case ONEW:
	case OPANIC:
	case OPRINT:
	case OPRINTN:
		fmtprint(f, "%#O(", n->op);
		if(n->left)
			exprfmt(f, n->left, 0);
		else
			exprlistfmt(f, n->list);
		fmtprint(f, ")");
		break;

	case OMAKESLICE:
		fmtprint(f, "make(%#T, ", n->type);
		exprfmt(f, n->left, 0);
		if(count(n->list) > 2) {
			fmtprint(f, ", ");
			exprfmt(f, n->right, 0);
		}
		fmtprint(f, ")");
		break;

	case OMAKEMAP:
		fmtprint(f, "make(%#T)", n->type);
		break;
	}

	if(prec > nprec)
		fmtprint(f, ")");
}
