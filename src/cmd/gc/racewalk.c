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

// TODO(dvyukov): do not instrument initialization as writes:
// a := make([]int, 10)

static void racewalklist(NodeList *l, NodeList **init);
static void racewalknode(Node **np, NodeList **init, int wr, int skip);
static int callinstr(Node **n, NodeList **init, int wr, int skip);
static Node* uintptraddr(Node *n);
static void makeaddable(Node *n);
static Node* basenod(Node *n);
static void foreach(Node *n, void(*f)(Node*, void*), void *c);
static void hascallspred(Node *n, void *c);
static void appendinit(Node **np, NodeList *init);
static Node* detachexpr(Node *n, NodeList **init);

// Do not instrument the following packages at all,
// at best instrumentation would cause infinite recursion.
static const char *omit_pkgs[] = {"runtime", "runtime/race"};
// Only insert racefuncenter/racefuncexit into the following packages.
// Memory accesses in the packages are either uninteresting or will cause false positives.
static const char *noinst_pkgs[] = {"sync", "sync/atomic"};

static int
ispkgin(const char **pkgs, int n)
{
	int i;

	if(myimportpath) {
		for(i=0; i<n; i++) {
			if(strcmp(myimportpath, pkgs[i]) == 0)
				return 1;
		}
	}
	return 0;
}

static int
isforkfunc(Node *fn)
{
	// Special case for syscall.forkAndExecInChild.
	// In the child, this function must not acquire any locks, because
	// they might have been locked at the time of the fork.  This means
	// no rescheduling, no malloc calls, and no new stack segments.
	// Race instrumentation does all of the above.
	return myimportpath != nil && strcmp(myimportpath, "syscall") == 0 &&
		strcmp(fn->nname->sym->name, "forkAndExecInChild") == 0;
}

void
racewalk(Node *fn)
{
	Node *nd;
	Node *nodpc;
	char s[1024];

	if(ispkgin(omit_pkgs, nelem(omit_pkgs)) || isforkfunc(fn))
		return;

	if(!ispkgin(noinst_pkgs, nelem(noinst_pkgs))) {
		racewalklist(fn->nbody, nil);
		// nothing interesting for race detector in fn->enter
		racewalklist(fn->exit, nil);
	}

	// nodpc is the PC of the caller as extracted by
	// getcallerpc. We use -widthptr(FP) for x86.
	// BUG: this will not work on arm.
	nodpc = nod(OXXX, nil, nil);
	*nodpc = *nodfp;
	nodpc->type = types[TUINTPTR];
	nodpc->xoffset = -widthptr;
	nd = mkcall("racefuncenter", T, nil, nodpc);
	fn->enter = concat(list1(nd), fn->enter);
	nd = mkcall("racefuncexit", T, nil);
	fn->exit = list(fn->exit, nd);

	if(debug['W']) {
		snprint(s, sizeof(s), "after racewalk %S", fn->nname->sym);
		dumplist(s, fn->nbody);
		snprint(s, sizeof(s), "enter %S", fn->nname->sym);
		dumplist(s, fn->enter);
		snprint(s, sizeof(s), "exit %S", fn->nname->sym);
		dumplist(s, fn->exit);
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
	NodeList *l;
	NodeList *fini;

	n = *np;

	if(n == N)
		return;

	if(debug['w'] > 1)
		dump("racewalk-before", n);
	setlineno(n);
	if(init == nil)
		fatal("racewalk: bad init list");
	if(init == &n->ninit) {
		// If init == &n->ninit and n->ninit is non-nil,
		// racewalknode might append it to itself.
		// nil it out and handle it separately before putting it back.
		l = n->ninit;
		n->ninit = nil;
		racewalklist(l, nil);
		racewalknode(&n, &l, wr, skip);  // recurse with nil n->ninit
		appendinit(&n, l);
		*np = n;
		return;
	}

	racewalklist(n->ninit, nil);

	switch(n->op) {
	default:
		fatal("racewalk: unknown node type %O", n->op);

	case OASOP:
	case OAS:
	case OAS2:
	case OAS2RECV:
	case OAS2FUNC:
	case OAS2MAPR:
		racewalknode(&n->left, init, 1, 0);
		racewalknode(&n->right, init, 0, 0);
		goto ret;

	case OCFUNC:
	case OVARKILL:
		// can't matter
		goto ret;

	case OBLOCK:
		if(n->list == nil)
			goto ret;

		switch(n->list->n->op) {
		case OCALLFUNC:
		case OCALLMETH:
		case OCALLINTER:
			// Blocks are used for multiple return function calls.
			// x, y := f() becomes BLOCK{CALL f, AS x [SP+0], AS y [SP+n]}
			// We don't want to instrument between the statements because it will
			// smash the results.
			racewalknode(&n->list->n, &n->list->n->ninit, 0, 0);
			fini = nil;
			racewalklist(n->list->next, &fini);
			n->list = concat(n->list, fini);
			break;

		default:
			// Ordinary block, for loop initialization or inlined bodies.
			racewalklist(n->list, nil);
			break;
		}
		goto ret;

	case ODEFER:
		racewalknode(&n->left, init, 0, 0);
		goto ret;

	case OPROC:
		racewalknode(&n->left, init, 0, 0);
		goto ret;

	case OCALLINTER:
		racewalknode(&n->left, init, 0, 0);
		goto ret;

	case OCALLFUNC:
		// Instrument dst argument of runtime.writebarrier* calls
		// as we do not instrument runtime code.
		if(n->left->sym != S && n->left->sym->pkg == runtimepkg && 
		(strncmp(n->left->sym->name, "writebarrier", 12) == 0 || strcmp(n->left->sym->name, "typedmemmove") == 0)) {
			// Find the dst argument.
			// The list can be reordered, so it's not necessary just the first or the second element.
			for(l = n->list; l; l = l->next) {
				if(strcmp(n->left->sym->name, "typedmemmove") == 0) {
					if(l->n->left->xoffset == widthptr)
						break;
				} else {
					if(l->n->left->xoffset == 0)
						break;
				}
			}
			if(l == nil)
				fatal("racewalk: writebarrier no arg");
			if(l->n->right->op != OADDR)
				fatal("racewalk: writebarrier bad arg");
			callinstr(&l->n->right->left, init, 1, 0);
		}
		racewalknode(&n->left, init, 0, 0);
		goto ret;

	case ONOT:
	case OMINUS:
	case OPLUS:
	case OREAL:
	case OIMAG:
	case OCOM:
		racewalknode(&n->left, init, wr, 0);
		goto ret;

	case ODOTINTER:
		racewalknode(&n->left, init, 0, 0);
		goto ret;

	case ODOT:
		racewalknode(&n->left, init, 0, 1);
		callinstr(&n, init, wr, skip);
		goto ret;

	case ODOTPTR: // dst = (*x).f with implicit *; otherwise it's ODOT+OIND
		racewalknode(&n->left, init, 0, 0);
		callinstr(&n, init, wr, skip);
		goto ret;

	case OIND: // *p
		racewalknode(&n->left, init, 0, 0);
		callinstr(&n, init, wr, skip);
		goto ret;

	case OSPTR:
	case OLEN:
	case OCAP:
		racewalknode(&n->left, init, 0, 0);
		if(istype(n->left->type, TMAP)) {
			n1 = nod(OCONVNOP, n->left, N);
			n1->type = ptrto(types[TUINT8]);
			n1 = nod(OIND, n1, N);
			typecheck(&n1, Erv);
			callinstr(&n1, init, 0, skip);
		}
		goto ret;

	case OLSH:
	case ORSH:
	case OLROT:
	case OAND:
	case OANDNOT:
	case OOR:
	case OXOR:
	case OSUB:
	case OMUL:
	case OHMUL:
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
		// walk has ensured the node has moved to a location where
		// side effects are safe.
		// n->right may not be executed,
		// so instrumentation goes to n->right->ninit, not init.
		racewalknode(&n->right, &n->right->ninit, wr, 0);
		goto ret;

	case ONAME:
		callinstr(&n, init, wr, skip);
		goto ret;

	case OCONV:
		racewalknode(&n->left, init, wr, 0);
		goto ret;

	case OCONVNOP:
		racewalknode(&n->left, init, wr, 0);
		goto ret;

	case ODIV:
	case OMOD:
		racewalknode(&n->left, init, wr, 0);
		racewalknode(&n->right, init, wr, 0);
		goto ret;

	case OINDEX:
		if(!isfixedarray(n->left->type))
			racewalknode(&n->left, init, 0, 0);
		else if(!islvalue(n->left)) {
			// index of unaddressable array, like Map[k][i].
			racewalknode(&n->left, init, wr, 0);
			racewalknode(&n->right, init, 0, 0);
			goto ret;
		}
		racewalknode(&n->right, init, 0, 0);
		if(n->left->type->etype != TSTRING)
			callinstr(&n, init, wr, skip);
		goto ret;

	case OSLICE:
	case OSLICEARR:
	case OSLICE3:
	case OSLICE3ARR:
		// Seems to only lead to double instrumentation.
		//racewalknode(&n->left, init, 0, 0);
		goto ret;

	case OADDR:
		racewalknode(&n->left, init, 0, 1);
		goto ret;

	case OEFACE:
		racewalknode(&n->left, init, 0, 0);
		racewalknode(&n->right, init, 0, 0);
		goto ret;

	case OITAB:
		racewalknode(&n->left, init, 0, 0);
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
	case OCONVIFACE:
	case OCMPIFACE:
	case OMAKECHAN:
	case OMAKEMAP:
	case OMAKESLICE:
	case OCALL:
	case OCOPY:
	case OAPPEND:
	case ORUNESTR:
	case OARRAYBYTESTR:
	case OARRAYRUNESTR:
	case OSTRARRAYBYTE:
	case OSTRARRAYRUNE:
	case OINDEXMAP:  // lowered to call
	case OCMPSTR:
	case OADDSTR:
	case ODOTTYPE:
	case ODOTTYPE2:
	case OAS2DOTTYPE:
	case OCALLPART: // lowered to PTRLIT
	case OCLOSURE:  // lowered to PTRLIT
	case ORANGE:    // lowered to ordinary for loop
	case OARRAYLIT: // lowered to assignments
	case OMAPLIT:
	case OSTRUCTLIT:
		yyerror("racewalk: %O must be lowered by now", n->op);
		goto ret;

	// impossible nodes: only appear in backend.
	case ORROTC:
	case OEXTEND:
		yyerror("racewalk: %O cannot exist now", n->op);
		goto ret;

	// just do generic traversal
	case OFOR:
	case OIF:
	case OCALLMETH:
	case ORETURN:
	case ORETJMP:
	case OSWITCH:
	case OSELECT:
	case OEMPTY:
	case OBREAK:
	case OCONTINUE:
	case OFALL:
	case OGOTO:
	case OLABEL:
		goto ret;

	// does not require instrumentation
	case OPRINT:     // don't bother instrumenting it
	case OPRINTN:    // don't bother instrumenting it
	case OCHECKNIL: // always followed by a read.
	case OPARAM:     // it appears only in fn->exit to copy heap params back
	case OCLOSUREVAR:// immutable pointer to captured variable
	case ODOTMETH:   // either part of CALLMETH or CALLPART (lowered to PTRLIT)
	case OINDREG:    // at this stage, only n(SP) nodes from nodarg
	case ODCL:       // declarations (without value) cannot be races
	case ODCLCONST:
	case ODCLTYPE:
	case OTYPE:
	case ONONAME:
	case OLITERAL:
	case OSLICESTR:  // always preceded by bounds checking, avoid double instrumentation.
	case OTYPESW:    // ignored by code generation, do not instrument.
		goto ret;
	}

ret:
	if(n->op != OBLOCK)  // OBLOCK is handled above in a special way.
		racewalklist(n->list, init);
	if(n->ntest != N)
		racewalknode(&n->ntest, &n->ntest->ninit, 0, 0);
	if(n->nincr != N)
		racewalknode(&n->nincr, &n->nincr->ninit, 0, 0);
	racewalklist(n->nbody, nil);
	racewalklist(n->nelse, nil);
	racewalklist(n->rlist, nil);
	*np = n;
}

static int
isartificial(Node *n)
{
	// compiler-emitted artificial things that we do not want to instrument,
	// cant' possibly participate in a data race.
	if(n->op == ONAME && n->sym != S && n->sym->name != nil) {
		if(strcmp(n->sym->name, "_") == 0)
			return 1;
		// autotmp's are always local
		if(strncmp(n->sym->name, "autotmp_", sizeof("autotmp_")-1) == 0)
			return 1;
		// statictmp's are read-only
		if(strncmp(n->sym->name, "statictmp_", sizeof("statictmp_")-1) == 0)
			return 1;
		// go.itab is accessed only by the compiler and runtime (assume safe)
		if(n->sym->pkg && n->sym->pkg->name && strcmp(n->sym->pkg->name, "go.itab") == 0)
			return 1;
	}
	return 0;
}

static int
callinstr(Node **np, NodeList **init, int wr, int skip)
{
	Node *f, *b, *n;
	Type *t;
	int class, hascalls;

	n = *np;
	//print("callinstr for %+N [ %O ] etype=%E class=%d\n",
	//	  n, n->op, n->type ? n->type->etype : -1, n->class);

	if(skip || n->type == T || n->type->etype >= TIDEAL)
		return 0;
	t = n->type;
	if(isartificial(n))
		return 0;

	b = basenod(n);
	// it skips e.g. stores to ... parameter array
	if(isartificial(b))
		return 0;
	class = b->class;
	// BUG: we _may_ want to instrument PAUTO sometimes
	// e.g. if we've got a local variable/method receiver
	// that has got a pointer inside. Whether it points to
	// the heap or not is impossible to know at compile time
	if((class&PHEAP) || class == PPARAMREF || class == PEXTERN
		|| b->op == OINDEX || b->op == ODOTPTR || b->op == OIND || b->op == OXDOT) {
		hascalls = 0;
		foreach(n, hascallspred, &hascalls);
		if(hascalls) {
			n = detachexpr(n, init);
			*np = n;
		}
		n = treecopy(n);
		makeaddable(n);
		if(t->etype == TSTRUCT || isfixedarray(t)) {
			f = mkcall(wr ? "racewriterange" : "racereadrange", T, init, uintptraddr(n),
					nodintconst(t->width));
		} else
			f = mkcall(wr ? "racewrite" : "raceread", T, init, uintptraddr(n));
		*init = list(*init, f);
		return 1;
	}
	return 0;
}

// makeaddable returns a node whose memory location is the
// same as n, but which is addressable in the Go language
// sense.
// This is different from functions like cheapexpr that may make
// a copy of their argument.
static void
makeaddable(Node *n)
{
	// The arguments to uintptraddr technically have an address but
	// may not be addressable in the Go sense: for example, in the case
	// of T(v).Field where T is a struct type and v is
	// an addressable value.
	switch(n->op) {
	case OINDEX:
		if(isfixedarray(n->left->type))
			makeaddable(n->left);
		break;
	case ODOT:
	case OXDOT:
		// Turn T(v).Field into v.Field
		if(n->left->op == OCONVNOP)
			n->left = n->left->left;
		makeaddable(n->left);
		break;
	case ODOTPTR:
	default:
		// nothing to do
		break;
	}
}

static Node*
uintptraddr(Node *n)
{
	Node *r;

	r = nod(OADDR, n, N);
	r->bounded = 1;
	r = conv(r, types[TUNSAFEPTR]);
	r = conv(r, types[TUINTPTR]);
	return r;
}

// basenod returns the simplest child node of n pointing to the same
// memory area.
static Node*
basenod(Node *n)
{
	for(;;) {
		if(n->op == ODOT || n->op == OXDOT || n->op == OCONVNOP || n->op == OCONV || n->op == OPAREN) {
			n = n->left;
			continue;
		}
		if(n->op == OINDEX && isfixedarray(n->type)) {
			n = n->left;
			continue;
		}
		break;
	}
	return n;
}

static Node*
detachexpr(Node *n, NodeList **init)
{
	Node *addr, *as, *ind, *l;

	addr = nod(OADDR, n, N);
	l = temp(ptrto(n->type));
	as = nod(OAS, l, addr);
	typecheck(&as, Etop);
	walkexpr(&as, init);
	*init = list(*init, as);
	ind = nod(OIND, l, N);
	typecheck(&ind, Erv);
	walkexpr(&ind, init);
	return ind;
}

static void
foreachnode(Node *n, void(*f)(Node*, void*), void *c)
{
	if(n)
		f(n, c);
}

static void
foreachlist(NodeList *l, void(*f)(Node*, void*), void *c)
{
	for(; l; l = l->next)
		foreachnode(l->n, f, c);
}

static void
foreach(Node *n, void(*f)(Node*, void*), void *c)
{
	foreachlist(n->ninit, f, c);
	foreachnode(n->left, f, c);
	foreachnode(n->right, f, c);
	foreachlist(n->list, f, c);
	foreachnode(n->ntest, f, c);
	foreachnode(n->nincr, f, c);
	foreachlist(n->nbody, f, c);
	foreachlist(n->nelse, f, c);
	foreachlist(n->rlist, f, c);
}

static void
hascallspred(Node *n, void *c)
{
	switch(n->op) {
	case OCALL:
	case OCALLFUNC:
	case OCALLMETH:
	case OCALLINTER:
		(*(int*)c)++;
	}
}

// appendinit is like addinit in subr.c
// but appends rather than prepends.
static void
appendinit(Node **np, NodeList *init)
{
	Node *n;

	if(init == nil)
		return;

	n = *np;
	switch(n->op) {
	case ONAME:
	case OLITERAL:
		// There may be multiple refs to this node;
		// introduce OCONVNOP to hold init list.
		n = nod(OCONVNOP, n, N);
		n->type = n->left->type;
		n->typecheck = 1;
		*np = n;
		break;
	}
	n->ninit = concat(n->ninit, init);
	n->ullman = UINF;
}

