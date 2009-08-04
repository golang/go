// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go.h"

enum
{
	PFIXME = 0,
	PCHAN = 0,
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

	nprec = 0;
	if(n == nil) {
		fmtprint(f, "<nil>");
		return;
	}

	switch(n->op) {
	case ONAME:
	case ONONAME:
	case OPACK:
	case OLITERAL:
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
	}

	if(prec > nprec)
		fmtprint(f, "(");

	switch(n->op) {
	default:
	bad:
		fmtprint(f, "(node %O)", n->op);
		break;

	case OLITERAL:
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
		fmtprint(f, "%T", n->type);
		break;

	case OTARRAY:
		fmtprint(f, "[]");
		exprfmt(f, n->left, PFIXME);
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
			exprfmt(f, n->left, PCHAN);
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

	case OCOMPLIT:
		fmtprint(f, "<compos>");
		break;

	case ODOT:
	case ODOTPTR:
	case ODOTINTER:
	case ODOTMETH:
		exprfmt(f, n->left, 7);
		if(n->right == N || n->right->sym == S)
			fmtprint(f, ".<nil>");
		else
			fmtprint(f, ".%s", n->right->sym->name);
		break;

	case ODOTTYPE:
		exprfmt(f, n->left, 7);
		fmtprint(f, ".(");
		exprfmt(f, n->right, 0);
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
		exprfmt(f, n->left, 7);
		fmtprint(f, "[");
		exprfmt(f, n->right->left, 0);
		fmtprint(f, ":");
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
		fmtprint(f, ")");
		break;

	case OCONV:
		fmtprint(f, "%T(", n->type);
		exprfmt(f, n->left, 0);
		fmtprint(f, ")");
		break;

	case OCAP:
	case OCLOSE:
	case OCLOSED:
	case OLEN:
	case OMAKE:
	case ONEW:
	case OPANIC:
	case OPANICN:
	case OPRINT:
	case OPRINTN:
		fmtprint(f, "%#O(", n->op);
		if(n->left)
			exprfmt(f, n->left, 0);
		else
			exprlistfmt(f, n->list);
		fmtprint(f, ")");
		break;
	}

	if(prec > nprec)
		fmtprint(f, ")");
}
