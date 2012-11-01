// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The racewalk pass modifies the code tree for the function as follows:
//
// 1. It inserts a call to racefuncenter at the beginning of each function.
// 2. It inserts a call to racefuncexit at the end of each function.
// 3. It inserts a call to raceread before each memory read.
// 4. It inserts a call to racewrite before each memory write.
//
// The rewriting is not yet complete. Certain nodes are not rewritten
// but should be.

#include <u.h>
#include <libc.h>
#include "go.h"
#include "opnames.h"

// TODO(dvyukov): do not instrument initialization as writes:
// a := make([]int, 10)

static void racewalklist(NodeList *l, NodeList **init);
static void racewalknode(Node **np, NodeList **init, int wr, int skip);
static int callinstr(Node *n, NodeList **init, int wr, int skip);
static Node* uintptraddr(Node *n);
static Node* basenod(Node *n);

static const char *omitPkgs[] = {"runtime", "runtime/race", "sync", "sync/atomic"};

void
racewalk(Node *fn)
{
	int i;
	Node *nd;
	Node *nodpc;
	char s[1024];

	if(myimportpath) {
		for(i=0; i<nelem(omitPkgs); i++) {
			if(strcmp(myimportpath, omitPkgs[i]) == 0)
				return;
		}
	}

	// nodpc is the PC of the caller as extracted by
	// getcallerpc. We use -widthptr(FP) for x86.
	// BUG: this will not work on arm.
	nodpc = nod(OXXX, nil, nil);
	*nodpc = *nodfp;
	nodpc->type = types[TUINTPTR];
	nodpc->xoffset = -widthptr;
	nd = mkcall("racefuncenter", T, nil, nodpc);
	fn->enter = list(fn->enter, nd);
	nd = mkcall("racefuncexit", T, nil);
	fn->exit = list(fn->exit, nd); // works fine if (!fn->exit)
	racewalklist(curfn->nbody, nil);

	if(debug['W']) {
		snprint(s, sizeof(s), "after racewalk %S", curfn->nname->sym);
		dumplist(s, curfn->nbody);
	}
}

static void
racewalklist(NodeList *l, NodeList **init)
{
	NodeList *instr;

	for(; l; l = l->next) {
		instr = nil;
		racewalknode(&l->n, &instr, 0, 0);
		if(init == nil)
			l->n->ninit = concat(l->n->ninit, instr);
		else
			*init = concat(*init, instr);
	}
}

// walkexpr and walkstmt combined
// walks the tree and adds calls to the
// instrumentation code to top-level (statement) nodes' init
static void
racewalknode(Node **np, NodeList **init, int wr, int skip)
{
	Node *n, *n1;

	n = *np;

	if(n == N)
		return;
	if(0)
		print("op=%s, left=[ %N ], right=[ %N ], right's type=%T, n's type=%T, n's class=%d\n",
			opnames[n->op], n->left, n->right, n->right ? n->right->type : nil, n->type, n->class);
	setlineno(n);

	racewalklist(n->ninit, nil);

	switch(n->op) {
	default:
		fatal("racewalk: unknown node type %O", n->op);

	case OASOP:
	case OAS:
	case OAS2:
	case OAS2DOTTYPE:
	case OAS2RECV:
	case OAS2FUNC:
	case OAS2MAPR:
		racewalknode(&n->left, init, 1, 0);
		racewalknode(&n->right, init, 0, 0);
		goto ret;

	case OBLOCK:
		// leads to crashes.
		//racewalklist(n->list, nil);
		goto ret;

	case ODEFER:
		racewalknode(&n->left, init, 0, 0);
		goto ret;

	case OFOR:
		if(n->ntest != N)
			racewalklist(n->ntest->ninit, nil);
		racewalknode(&n->nincr, init, wr, 0);
		racewalklist(n->nbody, nil);
		goto ret;

	case OIF:
		racewalknode(&n->ntest, &n->ninit, wr, 0);
		racewalklist(n->nbody, nil);
		racewalklist(n->nelse, nil);
		goto ret;

	case OPROC:
		racewalknode(&n->left, init, 0, 0);
		goto ret;

	case OCALLINTER:
		racewalknode(&n->left, init, 0, 0);
		racewalklist(n->list, init);
		goto ret;

	case OCALLFUNC:
		racewalknode(&n->left, init, 0, 0);
		racewalklist(n->list, init);
		goto ret;

	case OCALLMETH:
		racewalklist(n->list, init);
		goto ret;

	case ORETURN:
		racewalklist(n->list, nil);
		goto ret;

	case OSELECT:
		// n->nlist is nil by now because this code
		// is running after walkselect
		racewalklist(n->nbody, nil);
		goto ret;

	case OSWITCH:
		if(n->ntest->op == OTYPESW)
			// don't bother, we have static typization
			return;
		racewalknode(&n->ntest, &n->ninit, 0, 0);
		racewalklist(n->nbody, nil);
		goto ret;

	case OEMPTY:
		goto ret;

	case ONOT:
	case OMINUS:
	case OPLUS:
	case OREAL:
	case OIMAG:
		racewalknode(&n->left, init, wr, 0);
		goto ret;

	case ODOTINTER:
		racewalknode(&n->left, init, 0, 0);
		goto ret;

	case ODOT:
		callinstr(n, init, wr, skip);
		racewalknode(&n->left, init, 0, 1);
		goto ret;

	case ODOTPTR: // dst = (*x).f with implicit *; otherwise it's ODOT+OIND
		callinstr(n, init, wr, skip);
		racewalknode(&n->left, init, 0, 0);
		goto ret;

	case OIND: // *p
		callinstr(n, init, wr, skip);
		racewalknode(&n->left, init, 0, 0);
		goto ret;

	case OLEN:
	case OCAP:
		racewalknode(&n->left, init, 0, 0);
		if(istype(n->left->type, TMAP)) {
			// crashes on len(m[0]) or len(f())
			USED(n1);
			/*
			n1 = nod(OADDR, n->left, N);
			n1 = conv(n1, types[TUNSAFEPTR]);
			n1 = conv(n1, ptrto(ptrto(types[TINT8])));
			n1 = nod(OIND, n1, N);
			n1 = nod(OIND, n1, N);
			typecheck(&n1, Erv);
			callinstr(n1, init, 0, skip);
			*/
		}
		goto ret;

	case OLSH:
	case ORSH:
	case OAND:
	case OANDNOT:
	case OOR:
	case OXOR:
	case OSUB:
	case OMUL:
	case OEQ:
	case ONE:
	case OLT:
	case OLE:
	case OGE:
	case OGT:
	case OADD:
	case OCOMPLEX:
		racewalknode(&n->left, init, wr, 0);
		racewalknode(&n->right, init, wr, 0);
		goto ret;

	case OANDAND:
	case OOROR:
		racewalknode(&n->left, init, wr, 0);
		// It requires more complex tree transformation,
		// because we don't know whether it will be executed or not.
		//racewalknode(&n->right, init, wr, 0);
		goto ret;

	case ONAME:
		callinstr(n, init, wr, skip);
		goto ret;

	case OCONV:
		racewalknode(&n->left, init, wr, 0);
		goto ret;

	case OCONVNOP:
		racewalknode(&n->left, init, wr, 0);
		goto ret;

	case ODIV:
	case OMOD:
		// TODO(dvyukov): add a test for this
		racewalknode(&n->left, init, wr, 0);
		racewalknode(&n->right, init, wr, 0);
		goto ret;

	case OINDEX:
		if(n->left->type->etype != TSTRING)
			callinstr(n, init, wr, skip);
		if(!isfixedarray(n->left->type))
			racewalknode(&n->left, init, 0, 0);
		racewalknode(&n->right, init, 0, 0);
		goto ret;

	case OSLICE:
	case OSLICEARR:
		// Seems to only lead to double instrumentation.
		//racewalknode(&n->left, init, 0, 0);
		//racewalklist(n->list, init);
		goto ret;

	case OADDR:
		racewalknode(&n->left, init, 0, 1);
		goto ret;

	// should not appear in AST by now
	case OSEND:
	case ORECV:
	case OCLOSE:
	case ONEW:
	case OXCASE:
	case OXFALL:
	case OCASE:
	case OPANIC:
	case ORECOVER:
		yyerror("racewalk: %O must be lowered by now", n->op);
		goto ret;

	// does not require instrumentation
	case OINDEXMAP:  // implemented in runtime
	case OPRINT:  // don't bother instrumenting it
	case OPRINTN:  // don't bother instrumenting it
		goto ret;

	// unimplemented
	case OCMPSTR:
	case OADDSTR:
	case OSLICESTR:
	case OAPPEND:
	case OCOPY:
	case OMAKECHAN:
	case OMAKEMAP:
	case OMAKESLICE:
	case ORUNESTR:
	case OARRAYBYTESTR:
	case OARRAYRUNESTR:
	case OSTRARRAYBYTE:
	case OSTRARRAYRUNE:
	case OCMPIFACE:
	case OARRAYLIT:
	case OMAPLIT:
	case OSTRUCTLIT:
	case OCLOSURE:
	case ODOTTYPE:
	case ODOTTYPE2:
	case OCONVIFACE:
	case OCALL:
	case OBREAK:
	case ODCL:
	case OCONTINUE:
	case OFALL:
	case OGOTO:
	case OLABEL:
	case ODCLCONST:
	case ODCLTYPE:
	case OLITERAL:
	case ORANGE:
	case OTYPE:
	case ONONAME:
	case OINDREG:
	case OCOM:
	case ODOTMETH:
	case OEFACE:
	case OITAB:
	case OEXTEND:
	case OHMUL:
	case OLROT:
	case ORROTC:
		goto ret;
	}

ret:
	*np = n;
}

static int
callinstr(Node *n, NodeList **init, int wr, int skip)
{
	Node *f, *b;
	Type *t, *t1;
	int class, res;

	//print("callinstr for %N [ %s ] etype=%d class=%d\n",
	//	  n, opnames[n->op], n->type ? n->type->etype : -1, n->class);

	if(skip || n->type == T || n->type->etype >= TIDEAL)
		return 0;
	t = n->type;
	if(n->op == ONAME) {
		if(n->sym != S) {
			if(n->sym->name != nil) {
				if(strncmp(n->sym->name, "_", sizeof("_")-1) == 0)
					return 0;
				if(strncmp(n->sym->name, "autotmp_", sizeof("autotmp_")-1) == 0)
					return 0;
				if(strncmp(n->sym->name, "statictmp_", sizeof("statictmp_")-1) == 0)
					return 0;
			}
		}
	}
	if(t->etype == TSTRUCT) {
		res = 0;
		for(t1=t->type; t1; t1=t1->down) {
			if(t1->sym && strncmp(t1->sym->name, "_", sizeof("_")-1)) {
				n = treecopy(n);
				f = nod(OXDOT, n, newname(t1->sym));
				if(callinstr(f, init, wr, 0)) {
					typecheck(&f, Erv);
					res = 1;
				}
			}
		}
		return res;
	}

	b = basenod(n);
	class = b->class;
	// BUG: we _may_ want to instrument PAUTO sometimes
	// e.g. if we've got a local variable/method receiver
	// that has got a pointer inside. Whether it points to
	// the heap or not is impossible to know at compile time
	if((class&PHEAP) || class == PPARAMREF || class == PEXTERN
		|| b->type->etype == TARRAY || b->op == ODOTPTR || b->op == OIND || b->op == OXDOT) {
		n = treecopy(n);
		f = mkcall(wr ? "racewrite" : "raceread", T, nil, uintptraddr(n));
		//typecheck(&f, Etop);
		*init = list(*init, f);
		return 1;
	}
	return 0;
}

static Node*
uintptraddr(Node *n)
{
	Node *r;

	r = nod(OADDR, n, N);
	r = conv(r, types[TUNSAFEPTR]);
	r = conv(r, types[TUINTPTR]);
	return r;
}

static Node*
basenod(Node *n)
{
	for(;;) {
		if(n->op == ODOT || n->op == OPAREN) {
			n = n->left;
			continue;
		}
		if(n->op == OINDEX) {
			n = n->left;
			continue;
		}
		break;
	}
	return n;
}
