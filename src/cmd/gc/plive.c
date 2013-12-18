// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>
#include "gg.h"
#include "opt.h"
#include "../../pkg/runtime/funcdata.h"

enum { BitsPerPointer = 2 };

enum {
	UNVISITED = 0,
	VISITED = 1,
};

// An ordinary basic block.
//
// Instructions are threaded together in a doubly-linked list.  To iterate in
// program order follow the link pointer from the first node and stop after the
// last node has been visited
//
//   for(p = bb->first;; p = p->link) {
//     ...
//     if(p == bb->last)
//       break;
//   }
//
// To iterate in reverse program order by following the opt pointer from the
// last node
//
//   for(p = bb->last; p != nil; p = p->opt) {
//     ...
//   }
typedef struct BasicBlock BasicBlock;
struct BasicBlock {
	// An array of preceding blocks.  If the length of this array is 0 the
	// block is probably the start block of the CFG.
	Array *pred;

	// An array out succeeding blocks.  If the length of this array is zero,
	// the block probably ends in a return instruction.
	Array *succ;

	// First instruction in the block.  When part of a fully initialized
	// control flow graph, the opt member will be nil.
	Prog *first;

	// Last instruction in the basic block.
	Prog *last;

	// The reverse post order number.  This value is initialized to -1 and
	// will be replaced by a non-negative value when the CFG is constructed.
	// After CFG construction, if rpo is -1 this block is unreachable.
	int rpo;

	// State to denote whether the block has been visited during a
	// traversal.
	int mark;
};

// A collection of global state used by liveness analysis.
typedef struct Liveness Liveness;
struct Liveness {
	// A pointer to the node corresponding to the function being analyzed.
	Node *fn;

	// A linked list of instructions for this function.
	Prog *ptxt;

	// A list of arguments and local variables in this function.
	Array *vars;

	// A list of basic blocks that are overlayed on the instruction list.
	Array *cfg;

	// Summary sets of block effects.  The upward exposed variables and
	// variables killed sets are computed during the dataflow prologue.  The
	// live in and live out are solved for and serialized in the epilogue.
	Bvec **uevar;
	Bvec **varkill;
	Bvec **livein;
	Bvec **liveout;

	// An array with a bit vector for each safe point tracking live pointers
	// in the arguments and locals area.
	Array *argslivepointers;
	Array *livepointers;

	// An array with a bit vector for each safe point tracking dead values
	// pointers in the arguments and locals area.
	Array *argsdeadvalues;
	Array *deadvalues;
};

static int printnoise = 0;

static void*
xmalloc(uintptr size)
{
	void *result;

	result = malloc(size);
	if(result == nil)
		fatal("malloc failed");
	return result;
}

// Constructs a new basic block containing a single instruction.
static BasicBlock*
newblock(Prog *prog)
{
	BasicBlock *result;

	if(prog == nil)
		fatal("newblock: prog cannot be nil");
	result = xmalloc(sizeof(*result));
	result->rpo = -1;
	result->mark = UNVISITED;
	result->first = prog;
	result->last = prog;
	result->pred = arraynew(2, sizeof(BasicBlock*));
	result->succ = arraynew(2, sizeof(BasicBlock*));
	return result;
}

// Frees a basic block and all of its leaf data structures.
static void
freeblock(BasicBlock *bb)
{
	if(bb == nil)
		fatal("freeblock: cannot free nil");
	arrayfree(bb->pred);
	arrayfree(bb->succ);
	free(bb);
}

// Adds an edge between two basic blocks by making from a predecessor of to and
// to a successor of from.
static void
addedge(BasicBlock *from, BasicBlock *to)
{
	if(from == nil)
		fatal("addedge: from is nil");
	if(to == nil)
		fatal("addedge: to is nil");
	arrayadd(from->succ, &to);
	arrayadd(to->pred, &from);
}

// Inserts a new instruction ahead of an existing instruction in the instruction
// stream.  Any control flow, such as branches or fall throughs, that target the
// existing instruction are adjusted to target the new instruction.
static void
splicebefore(Liveness *lv, BasicBlock *bb, Prog *prev, Prog *curr)
{
	Prog *p;

	prev->opt = curr->opt;
	curr->opt = prev;
	prev->link = curr;
	if(prev->opt != nil)
		((Prog*)prev->opt)->link = prev;
	else
		bb->first = prev;
	for(p = lv->ptxt; p != nil; p = p->link) {
		if(p != prev) {
			if(p->link == curr)
				p->link = prev;
			if(p->as != ACALL && p->to.type == D_BRANCH && p->to.u.branch == curr)
				p->to.u.branch = prev;
		}
	}
}

// A pretty printer for basic blocks.
static void
printblock(BasicBlock *bb)
{
	BasicBlock *pred;
	BasicBlock *succ;
	Prog *prog;
	int i;

	print("basic block %d\n", bb->rpo);
	print("\tpred:");
	for(i = 0; i < arraylength(bb->pred); i++) {
		pred = *(BasicBlock**)arrayget(bb->pred, i);
		print(" %d", pred->rpo);
	}
	print("\n");
	print("\tsucc:");
	for(i = 0; i < arraylength(bb->succ); i++) {
		succ = *(BasicBlock**)arrayget(bb->succ, i);
		print(" %d", succ->rpo);
	}
	print("\n");
	print("\tprog:\n");
	for(prog = bb->first;; prog=prog->link) {
		print("\t\t%P\n", prog);
		if(prog == bb->last)
			break;
	}
}


// Iterates over a basic block applying a callback to each instruction.  There
// are two criteria for termination.  If the end of basic block is reached a
// value of zero is returned.  If the callback returns a non-zero value, the
// iteration is stopped and the value of the callback is returned.
static int
blockany(BasicBlock *bb, int (*callback)(Prog*))
{
	Prog *p;
	int result;

	for(p = bb->last; p != nil; p = p->opt) {
		result = (*callback)(p);
		if(result != 0)
			return result;
	}
	return 0;
}

// Collects and returns and array of Node*s for functions arguments and local
// variables.  TODO(cshapiro): only return pointer containing nodes if we are
// not also generating a dead value map.
static Array*
getvariables(Node *fn)
{
	Array *result;
	NodeList *ll;

	result = arraynew(0, sizeof(Node*));
	for(ll = fn->dcl; ll != nil; ll = ll->next) {
		if(ll->n->op == ONAME) {
			switch(ll->n->class & ~PHEAP) {
			case PAUTO:
			case PPARAM:
			case PPARAMOUT:
				arrayadd(result, &ll->n);
			}
		}
	}
	return result;
}

// A pretty printer for control flow graphs.  Takes an array of BasicBlock*s.
static void
printcfg(Array *cfg)
{
	BasicBlock *bb;
	int32 i;

	for(i = 0; i < arraylength(cfg); i++) {
		bb = *(BasicBlock**)arrayget(cfg, i);
		printblock(bb);
	}
}

// Assigns a reverse post order number to each connected basic block using the
// standard algorithm.  Unconnected blocks will not be affected.
static void
reversepostorder(BasicBlock *root, int32 *rpo)
{
	BasicBlock *bb;
	int i;

	root->mark = VISITED;
	for(i = 0; i < arraylength(root->succ); i++) {
		bb = *(BasicBlock**)arrayget(root->succ, i);
		if(bb->mark == UNVISITED)
			reversepostorder(bb, rpo);
	}
	*rpo -= 1;
	root->rpo = *rpo;
}

// Comparison predicate used for sorting basic blocks by their rpo in ascending
// order.
static int
blockrpocmp(const void *p1, const void *p2)
{
	BasicBlock *bb1;
	BasicBlock *bb2;

	bb1 = *(BasicBlock**)p1;
	bb2 = *(BasicBlock**)p2;
	if(bb1->rpo < bb2->rpo)
		return -1;
	if(bb1->rpo > bb2->rpo)
		return 1;
	return 0;
}

// A pattern matcher for call instructions.  Returns true when the instruction
// is a call to a specific package qualified function name.
static int
iscall(Prog *prog, LSym *name)
{
	if(prog == nil)
		fatal("iscall: prog is nil");
	if(name == nil)
		fatal("iscall: function name is nil");
	if(prog->as != ACALL)
		return 0;
	return name == prog->to.sym;
}

// Returns true for instructions that call a runtime function implementing a
// select communication clause.
static int
isselectcommcasecall(Prog *prog)
{
	static LSym* names[5];
	int32 i;

	if(names[0] == nil) {
		names[0] = linksym(pkglookup("selectsend", runtimepkg));
		names[1] = linksym(pkglookup("selectrecv", runtimepkg));
		names[2] = linksym(pkglookup("selectrecv2", runtimepkg));
		names[3] = linksym(pkglookup("selectdefault", runtimepkg));
	}
	for(i = 0; names[i] != nil; i++)
		if(iscall(prog, names[i]))
			return 1;
	return 0;
}

// Returns true for call instructions that target runtime·newselect.
static int
isnewselect(Prog *prog)
{
	static LSym *sym;

	if(sym == nil)
		sym = linksym(pkglookup("newselect", runtimepkg));
	return iscall(prog, sym);
}

// Returns true for call instructions that target runtime·selectgo.
static int
isselectgocall(Prog *prog)
{
	static LSym *sym;

	if(sym == nil)
		sym = linksym(pkglookup("selectgo", runtimepkg));
	return iscall(prog, sym);
}

static int
isdeferreturn(Prog *prog)
{
	static LSym *sym;

	if(sym == nil)
		sym = linksym(pkglookup("deferreturn", runtimepkg));
	return iscall(prog, sym);
}

// Walk backwards from a runtime·selectgo call up to its immediately dominating
// runtime·newselect call.  Any successor nodes of communication clause nodes
// are implicit successors of the runtime·selectgo call node.  The goal of this
// analysis is to add these missing edges to complete the control flow graph.
static void
addselectgosucc(BasicBlock *selectgo)
{
	BasicBlock *pred;
	BasicBlock *succ;

	pred = selectgo;
	for(;;) {
		if(arraylength(pred->pred) == 0)
			fatal("selectgo does not have a newselect");
		pred = *(BasicBlock**)arrayget(pred->pred, 0);
		if(blockany(pred, isselectcommcasecall)) {
			// A select comm case block should have exactly one
			// successor.
			if(arraylength(pred->succ) != 1)
				fatal("select comm case has too many successors");
			succ = *(BasicBlock**)arrayget(pred->succ, 0);
			// Its successor should have exactly two successors.
			// The drop through should flow to the selectgo block
			// and the branch should lead to the select case
			// statements block.
			if(arraylength(succ->succ) != 2)
				fatal("select comm case successor has too many successors");
			// Add the block as a successor of the selectgo block.
			addedge(selectgo, succ);
		}
		if(blockany(pred, isnewselect)) {
			// Reached the matching newselect.
			break;
		}
	}
}

// The entry point for the missing selectgo control flow algorithm.  Takes an
// array of BasicBlock*s containing selectgo calls.
static void
fixselectgo(Array *selectgo)
{
	BasicBlock *bb;
	int32 i;

	for(i = 0; i < arraylength(selectgo); i++) {
		bb = *(BasicBlock**)arrayget(selectgo, i);
		addselectgosucc(bb);
	}
}

// Constructs a control flow graph from a sequence of instructions.  This
// procedure is complicated by various sources of implicit control flow that are
// not accounted for using the standard cfg construction algorithm.  Returns an
// array of BasicBlock*s in control flow graph form (basic blocks ordered by
// their RPO number).
static Array*
newcfg(Prog *firstp)
{
	Prog *p;
	Prog *prev;
	BasicBlock *bb;
	Array *cfg;
	Array *selectgo;
	int32 i;
	int32 rpo;

	// Reset the opt field of each prog to nil.  In the first and second
	// passes, instructions that are labels temporarily use the opt field to
	// point to their basic block.  In the third pass, the opt field reset
	// to point to the predecessor of an instruction in its basic block.
	for(p = firstp; p != P; p = p->link)
		p->opt = nil;

	// Allocate an array to remember where we have seen selectgo calls.
	// These blocks will be revisited to add successor control flow edges.
	selectgo = arraynew(0, sizeof(BasicBlock*));

	// Loop through all instructions identifying branch targets
	// and fall-throughs and allocate basic blocks.
	cfg = arraynew(0, sizeof(BasicBlock*));
	bb = newblock(firstp);
	arrayadd(cfg, &bb);
	for(p = firstp; p != P; p = p->link) {
		if(p->to.type == D_BRANCH) {
			if(p->to.u.branch == nil)
				fatal("prog branch to nil");
			if(p->to.u.branch->opt == nil) {
				p->to.u.branch->opt = newblock(p->to.u.branch);
				arrayadd(cfg, &p->to.u.branch->opt);
			}
			if(p->as != AJMP && p->link != nil && p->link->opt == nil) {
				p->link->opt = newblock(p->link);
				arrayadd(cfg, &p->link->opt);
			}
		} else if(isselectcommcasecall(p) || isselectgocall(p)) {
			// Accommodate implicit selectgo control flow.
			if(p->link->opt == nil) {
				p->link->opt = newblock(p->link);
				arrayadd(cfg, &p->link->opt);
			}
		}
	}

	// Loop through all basic blocks maximally growing the list of
	// contained instructions until a label is reached.  Add edges
	// for branches and fall-through instructions.
	for(i = 0; i < arraylength(cfg); i++) {
		bb = *(BasicBlock**)arrayget(cfg, i);
		for(p = bb->last; p != nil; p = p->link) {
			if(p->opt != nil && p != bb->last)
				break;
			bb->last = p;

			// Pattern match an unconditional branch followed by a
			// dead return instruction.  This avoids a creating
			// unreachable control flow nodes.
			if(p->link != nil && p->link->link == nil)
				if (p->as == AJMP && p->link->as == ARET && p->link->opt == nil)
					break;

			// Collect basic blocks with selectgo calls.
			if(isselectgocall(p))
				arrayadd(selectgo, &bb);
		}
		if(bb->last->to.type == D_BRANCH)
			addedge(bb, bb->last->to.u.branch->opt);
		if(bb->last->link != nil) {
			// Add a fall-through when the instruction is
			// not an unconditional control transfer.
			switch(bb->last->as) {
			case AJMP:
			case ARET:
				break;
			default:
				addedge(bb, bb->last->link->opt);
			}
		}
	}

	// Add back links so the instructions in a basic block can be traversed
	// backward.  This is the final state of the instruction opt field.
	for(i = 0; i < arraylength(cfg); i++) {
		bb = *(BasicBlock**)arrayget(cfg, i);
		p = bb->first;
		prev = nil;
		for(;;) {
			p->opt = prev;
			if(p == bb->last)
				break;
			prev = p;
			p = p->link;
		}
	}

	// Add missing successor edges to the selectgo blocks.
	if(arraylength(selectgo))
		fixselectgo(selectgo);
	arrayfree(selectgo);

	// Find a depth-first order and assign a depth-first number to
	// all basic blocks.
	for(i = 0; i < arraylength(cfg); i++) {
		bb = *(BasicBlock**)arrayget(cfg, i);
		bb->mark = UNVISITED;
	}
	bb = *(BasicBlock**)arrayget(cfg, 0);
	rpo = arraylength(cfg);
	reversepostorder(bb, &rpo);

	// Sort the basic blocks by their depth first number.  The
	// array is now a depth-first spanning tree with the first
	// node being the root.
	arraysort(cfg, blockrpocmp);
	bb = *(BasicBlock**)arrayget(cfg, 0);

	// Unreachable control flow nodes are indicated by a -1 in the rpo
	// field.  If we see these nodes something must have gone wrong in an
	// upstream compilation phase.
	if(bb->rpo == -1)
		fatal("newcfg: unreferenced basic blocks");

	return cfg;
}

// Frees a control flow graph (an array of BasicBlock*s) and all of its leaf
// data structures.
static void
freecfg(Array *cfg)
{
	BasicBlock *bb;
	BasicBlock *bb0;
	Prog *p;
	int32 i;
	int32 len;

	len = arraylength(cfg);
	if(len > 0) {
		bb0 = *(BasicBlock**)arrayget(cfg, 0);
		for(p = bb0->first; p != P; p = p->link) {
			p->opt = nil;
		}
		for(i = 0; i < len; i++) {
			bb = *(BasicBlock**)arrayget(cfg, i);
			freeblock(bb);
		}
	}
	arrayfree(cfg);
}

// Returns true if the node names a variable that is otherwise uninteresting to
// the liveness computation.
static int
isfunny(Node *node)
{
	char *names[] = { ".fp", ".args", "_", nil };
	int i;

	if(node->sym != nil && node->sym->name != nil)
		for(i = 0; names[i] != nil; i++)
			if(strcmp(node->sym->name, names[i]) == 0)
				return 1;
	return 0;
}

// Computes the upward exposure and kill effects of an instruction on a set of
// variables.  The vars argument is an array of Node*s.
static void
progeffects(Prog *prog, Array *vars, Bvec *uevar, Bvec *varkill)
{
	ProgInfo info;
	Adr *from;
	Adr *to;
	Node *node;
	int32 i;
	int32 pos;

	bvresetall(uevar);
	bvresetall(varkill);
	proginfo(&info, prog);
	if(prog->as == ARET) {
		// Return instructions implicitly read all the arguments.  For
		// the sake of correctness, out arguments must be read.  For the
		// sake of backtrace quality, we read in arguments as well.
		for(i = 0; i < arraylength(vars); i++) {
			node = *(Node**)arrayget(vars, i);
			switch(node->class & ~PHEAP) {
			case PPARAM:
			case PPARAMOUT:
				bvset(uevar, i);
				break;
			case PAUTO:
				// Because the lifetime of stack variables
				// that have their address taken is undecidable,
				// we conservatively assume their lifetime
				// extends to the return as well.
				if(isfat(node->type) || node->addrtaken)
					bvset(uevar, i);
			}
		}
		return;
	}
	if(prog->as == ATEXT) {
		// A text instruction marks the entry point to a function and
		// the definition point of all in arguments.
		for(i = 0; i < arraylength(vars); i++) {
			node = *(Node**)arrayget(vars, i);
			switch(node->class & ~PHEAP) {
			case PPARAM:
				bvset(varkill, i);
			}
		}
		return;
	}
	if(info.flags & (LeftRead | LeftWrite | LeftAddr)) {
		from = &prog->from;
		if (from->node != nil && !isfunny(from->node) && from->sym != nil) {
			switch(prog->from.node->class & ~PHEAP) {
			case PAUTO:
			case PPARAM:
			case PPARAMOUT:
				pos = arrayindexof(vars, from->node);
				if(pos == -1)
					fatal("progeffects: variable %N is unknown", prog->from.node);
				if(info.flags & (LeftRead | LeftAddr))
					bvset(uevar, pos);
				if(info.flags & LeftWrite)
					if(from->node != nil && (!isfat(from->node->type) || prog->as == AFATVARDEF))
						bvset(varkill, pos);
			}
		}
	}
	if(info.flags & (RightRead | RightWrite | RightAddr)) {
		to = &prog->to;
		if (to->node != nil && to->sym != nil && !isfunny(to->node)) {
			switch(prog->to.node->class & ~PHEAP) {
			case PAUTO:
			case PPARAM:
			case PPARAMOUT:
				pos = arrayindexof(vars, to->node);
				if(pos == -1)
					fatal("progeffects: variable %N is unknown", to->node);
				if(info.flags & (RightRead | RightAddr))
					bvset(uevar, pos);
				if(info.flags & RightWrite)
					if(to->node != nil && (!isfat(to->node->type) || prog->as == AFATVARDEF))
						bvset(varkill, pos);
			}
		}
	}
}

// Constructs a new liveness structure used to hold the global state of the
// liveness computation.  The cfg argument is an array of BasicBlock*s and the
// vars argument is an array of Node*s.
static Liveness*
newliveness(Node *fn, Prog *ptxt, Array *cfg, Array *vars)
{
	Liveness *result;
	int32 i;
	int32 nblocks;
	int32 nvars;

	result = xmalloc(sizeof(*result));
	result->fn = fn;
	result->ptxt = ptxt;
	result->cfg = cfg;
	result->vars = vars;

	nblocks = arraylength(cfg);
	result->uevar = xmalloc(sizeof(Bvec*) * nblocks);
	result->varkill = xmalloc(sizeof(Bvec*) * nblocks);
	result->livein = xmalloc(sizeof(Bvec*) * nblocks);
	result->liveout = xmalloc(sizeof(Bvec*) * nblocks);

	nvars = arraylength(vars);
	for(i = 0; i < nblocks; i++) {
		result->uevar[i] = bvalloc(nvars);
		result->varkill[i] = bvalloc(nvars);
		result->livein[i] = bvalloc(nvars);
		result->liveout[i] = bvalloc(nvars);
	}

	result->livepointers = arraynew(0, sizeof(Bvec*));
	result->argslivepointers = arraynew(0, sizeof(Bvec*));
	result->deadvalues = arraynew(0, sizeof(Bvec*));
	result->argsdeadvalues = arraynew(0, sizeof(Bvec*));
	return result;
}

// Frees the liveness structure and all of its leaf data structures.
static void
freeliveness(Liveness *lv)
{
	int32 i;

	if(lv == nil)
		fatal("freeliveness: cannot free nil");

	for(i = 0; i < arraylength(lv->livepointers); i++)
		free(*(Bvec**)arrayget(lv->livepointers, i));
	arrayfree(lv->livepointers);

	for(i = 0; i < arraylength(lv->argslivepointers); i++)
		free(*(Bvec**)arrayget(lv->argslivepointers, i));
	arrayfree(lv->argslivepointers);

	for(i = 0; i < arraylength(lv->deadvalues); i++)
		free(*(Bvec**)arrayget(lv->deadvalues, i));
	arrayfree(lv->deadvalues);

	for(i = 0; i < arraylength(lv->argsdeadvalues); i++)
		free(*(Bvec**)arrayget(lv->argsdeadvalues, i));
	arrayfree(lv->argsdeadvalues);

	for(i = 0; i < arraylength(lv->cfg); i++) {
		free(lv->uevar[i]);
		free(lv->varkill[i]);
		free(lv->livein[i]);
		free(lv->liveout[i]);
	}

	free(lv->uevar);
	free(lv->varkill);
	free(lv->livein);
	free(lv->liveout);

	free(lv);
}

static void
printeffects(Prog *p, Bvec *uevar, Bvec *varkill)
{
	print("effects of %P", p);
	print("\nuevar: ");
	bvprint(uevar);
	print("\nvarkill: ");
	bvprint(varkill);
	print("\n");
}

// Pretty print a variable node.  Uses Pascal like conventions for pointers and
// addresses to avoid confusing the C like conventions used in the node variable
// names.
static void
printnode(Node *node)
{
	char *p;
	char *a;

	p = haspointers(node->type) ? "^" : "";
	a = node->addrtaken ? "@" : "";
	print(" %N%s%s", node, p, a);
}

// Pretty print a list of variables.  The vars argument is an array of Node*s.
static void
printvars(char *name, Bvec *bv, Array *vars)
{
	int32 i;

	print("%s:", name);
	for(i = 0; i < arraylength(vars); i++)
		if(bvget(bv, i))
			printnode(*(Node**)arrayget(vars, i));
	print("\n");
}

// Prints a basic block annotated with the information computed by liveness
// analysis.
static void
livenessprintblock(Liveness *lv, BasicBlock *bb)
{
	BasicBlock *pred;
	BasicBlock *succ;
	Prog *prog;
	Bvec *live;
	int i;
	int32 pos;

	print("basic block %d\n", bb->rpo);

	print("\tpred:");
	for(i = 0; i < arraylength(bb->pred); i++) {
		pred = *(BasicBlock**)arrayget(bb->pred, i);
		print(" %d", pred->rpo);
	}
	print("\n");

	print("\tsucc:");
	for(i = 0; i < arraylength(bb->succ); i++) {
		succ = *(BasicBlock**)arrayget(bb->succ, i);
		print(" %d", succ->rpo);
	}
	print("\n");

	printvars("\tuevar", lv->uevar[bb->rpo], lv->vars);
	printvars("\tvarkill", lv->varkill[bb->rpo], lv->vars);
	printvars("\tlivein", lv->livein[bb->rpo], lv->vars);
	printvars("\tliveout", lv->liveout[bb->rpo], lv->vars);

	print("\tprog:\n");
	for(prog = bb->first;; prog = prog->link) {
		print("\t\t%P", prog);
		if(prog->as == APCDATA && prog->from.offset == PCDATA_StackMapIndex) {
			pos = prog->to.offset;
			live = *(Bvec**)arrayget(lv->livepointers, pos);
			print(" ");
			bvprint(live);
		}
		print("\n");
		if(prog == bb->last)
			break;
	}
}

// Prints a control flow graph annotated with any information computed by
// liveness analysis.
static void
livenessprintcfg(Liveness *lv)
{
	BasicBlock *bb;
	int32 i;

	for(i = 0; i < arraylength(lv->cfg); i++) {
		bb = *(BasicBlock**)arrayget(lv->cfg, i);
		livenessprintblock(lv, bb);
	}
}

static void
checkauto(Node *fn, Prog *p, Node *n, char *where)
{
	NodeList *ll;
	int found;
	char *fnname;
	char *nname;

	found = 0;
	for(ll = fn->dcl; ll != nil; ll = ll->next) {
		if(ll->n->op == ONAME && ll->n->class == PAUTO) {
			if(n == ll->n) {
				found = 1;
				break;
			}
		}
	}
	if(found)
		return;
	fnname = fn->nname->sym->name ? fn->nname->sym->name : "<unknown>";
	nname = n->sym->name ? n->sym->name : "<unknown>";
	print("D_AUTO '%s' not found: name is '%s' function is '%s' class is %d\n", where, nname, fnname, n->class);
	print("Here '%P'\nlooking for node %p\n", p, n);
	for(ll = fn->dcl; ll != nil; ll = ll->next)
		print("node=%p, node->class=%d\n", (uintptr)ll->n, ll->n->class);
	yyerror("checkauto: invariant lost");
}

static void
checkparam(Node *fn, Prog *p, Node *n, char *where)
{
	NodeList *ll;
	int found;
	char *fnname;
	char *nname;

	if(isfunny(n))
		return;
	found = 0;
	for(ll = fn->dcl; ll != nil; ll = ll->next) {
		if(ll->n->op == ONAME && ((ll->n->class & ~PHEAP) == PPARAM ||
					  (ll->n->class & ~PHEAP) == PPARAMOUT)) {
			if(n == ll->n) {
				found = 1;
				break;
			}
		}
	}
	if(found)
		return;
	if(n->sym) {
		fnname = fn->nname->sym->name ? fn->nname->sym->name : "<unknown>";
		nname = n->sym->name ? n->sym->name : "<unknown>";
		print("D_PARAM '%s' not found: name='%s' function='%s' class is %d\n", where, nname, fnname, n->class);
		print("Here '%P'\nlooking for node %p\n", p, n);
		for(ll = fn->dcl; ll != nil; ll = ll->next)
			print("node=%p, node->class=%d\n", ll->n, ll->n->class);
	}
	yyerror("checkparam: invariant lost");
}

static void
checkprog(Node *fn, Prog *p)
{
	if(p->from.type == D_AUTO)
		checkauto(fn, p, p->from.node, "from");
	if(p->from.type == D_PARAM)
		checkparam(fn, p, p->from.node, "from");
	if(p->to.type == D_AUTO)
		checkauto(fn, p, p->to.node, "to");
	if(p->to.type == D_PARAM)
		checkparam(fn, p, p->to.node, "to");
}

// Check instruction invariants.  We assume that the nodes corresponding to the
// sources and destinations of memory operations will be declared in the
// function.  This is not strictly true, as is the case for the so-called funny
// nodes and there are special cases to skip over that stuff.  The analysis will
// fail if this invariant blindly changes.
static void
checkptxt(Node *fn, Prog *firstp)
{
	Prog *p;

	for(p = firstp; p != P; p = p->link) {
		if(0)
			print("analyzing '%P'\n", p);
		switch(p->as) {
		case ADATA:
		case AGLOBL:
		case ANAME:
		case ASIGNAME:
		case ATYPE:
			continue;
		}
		checkprog(fn, p);
	}
}

static void
twobitwalktype1(Type *t, vlong *xoffset, Bvec *bv)
{
	vlong fieldoffset;
	vlong i;
	vlong o;
	Type *t1;

	if(t->align > 0 && (*xoffset % t->align) != 0)
		fatal("twobitwalktype1: invalid initial alignment, %T", t);

	switch(t->etype) {
	case TINT8:
	case TUINT8:
	case TINT16:
	case TUINT16:
	case TINT32:
	case TUINT32:
	case TINT64:
	case TUINT64:
	case TINT:
	case TUINT:
	case TUINTPTR:
	case TBOOL:
	case TFLOAT32:
	case TFLOAT64:
	case TCOMPLEX64:
	case TCOMPLEX128:
		*xoffset += t->width;
		break;

	case TPTR32:
	case TPTR64:
	case TUNSAFEPTR:
	case TFUNC:
	case TCHAN:
	case TMAP:
		if(*xoffset % widthptr != 0)
			fatal("twobitwalktype1: invalid alignment, %T", t);
		bvset(bv, (*xoffset / widthptr) * BitsPerPointer);
		*xoffset += t->width;
		break;

	case TSTRING:
		// struct { byte *str; intgo len; }
		if(*xoffset % widthptr != 0)
			fatal("twobitwalktype1: invalid alignment, %T", t);
		bvset(bv, (*xoffset / widthptr) * BitsPerPointer);
		*xoffset += t->width;
		break;

	case TINTER:
		// struct { Itab *tab;	union { void *ptr, uintptr val } data; }
		// or, when isnilinter(t)==true:
		// struct { Type *type; union { void *ptr, uintptr val } data; }
		if(*xoffset % widthptr != 0)
			fatal("twobitwalktype1: invalid alignment, %T", t);
		bvset(bv, ((*xoffset / widthptr) * BitsPerPointer) + 1);
		if(isnilinter(t))
			bvset(bv, ((*xoffset / widthptr) * BitsPerPointer));
		*xoffset += t->width;
		break;

	case TARRAY:
		// The value of t->bound is -1 for slices types and >0 for
		// for fixed array types.  All other values are invalid.
		if(t->bound < -1)
			fatal("twobitwalktype1: invalid bound, %T", t);
		if(isslice(t)) {
			// struct { byte *array; uintgo len; uintgo cap; }
			if(*xoffset % widthptr != 0)
				fatal("twobitwalktype1: invalid TARRAY alignment, %T", t);
			bvset(bv, (*xoffset / widthptr) * BitsPerPointer);
			*xoffset += t->width;
		} else if(!haspointers(t->type))
				*xoffset += t->width;
		else
			for(i = 0; i < t->bound; i++)
				twobitwalktype1(t->type, xoffset, bv);
		break;

	case TSTRUCT:
		o = 0;
		for(t1 = t->type; t1 != T; t1 = t1->down) {
			fieldoffset = t1->width;
			*xoffset += fieldoffset - o;
			twobitwalktype1(t1->type, xoffset, bv);
			o = fieldoffset + t1->type->width;
		}
		*xoffset += t->width - o;
		break;

	default:
		fatal("twobitwalktype1: unexpected type, %T", t);
	}
}

// Returns the number of words of local variables.
static int32
localswords(void)
{
	return stkptrsize / widthptr;
}

// Returns the number of words of in and out arguments.
static int32
argswords(void)
{
	return curfn->type->argwid / widthptr;
}

// Generates live pointer value maps for arguments and local variables.  The
// this argument and the in arguments are always assumed live.  The vars
// argument is an array of Node*s.
static void
twobitlivepointermap(Liveness *lv, Bvec *liveout, Array *vars, Bvec *args, Bvec *locals)
{
	Node *node;
	Type *thisargtype;
	Type *inargtype;
	vlong xoffset;
	int32 i;

	for(i = 0; i < arraylength(vars); i++) {
		node = *(Node**)arrayget(vars, i);
		switch(node->class) {
		case PAUTO:
			if(bvget(liveout, i) && haspointers(node->type)) {
				xoffset = node->xoffset + stkptrsize;
				twobitwalktype1(node->type, &xoffset, locals);
			}
			break;
		case PPARAM:
		case PPARAMOUT:
			if(bvget(liveout, i) && haspointers(node->type)) {
				xoffset = node->xoffset;
				twobitwalktype1(node->type, &xoffset, args);
			}
			break;
		}
	}
	// In various and obscure circumstances, such as methods with an unused
	// receiver, the this argument and in arguments are omitted from the
	// node list.  We must explicitly preserve these values to ensure that
	// the addresses printed in backtraces are valid.
	thisargtype = getthisx(lv->fn->type);
	if(thisargtype != nil) {
		xoffset = 0;
		twobitwalktype1(thisargtype, &xoffset, args);
	}
	inargtype = getinargx(lv->fn->type);
	if(inargtype != nil) {
		xoffset = 0;
		twobitwalktype1(inargtype, &xoffset, args);
	}
}


// Generates dead value maps for arguments and local variables.  Dead values of
// any type are tracked, not just pointers.  The this argument and the in
// arguments are never assumed dead.  The vars argument is an array of Node*s.
static void
twobitdeadvaluemap(Liveness *lv, Bvec *liveout, Array *vars, Bvec *args, Bvec *locals)
{
	Node *node;
	/*
	Type *thisargtype;
	Type *inargtype;
	*/
	vlong xoffset;
	int32 i;

	for(i = 0; i < arraylength(vars); i++) {
		node = *(Node**)arrayget(vars, i);
		switch(node->class) {
		case PAUTO:
			if(!bvget(liveout, i)) {
				xoffset = node->xoffset + stkptrsize;
				twobitwalktype1(node->type, &xoffset, locals);
			}
			break;
		case PPARAM:
		case PPARAMOUT:
			if(!bvget(liveout, i)) {
				xoffset = node->xoffset;
				twobitwalktype1(node->type, &xoffset, args);
			}
			break;
		}
	}
	USED(lv);
	/*
	thisargtype = getinargx(lv->fn->type);
	if(thisargtype != nil) {
		xoffset = 0;
		twobitwalktype1(thisargtype, &xoffset, args);
	}
	inargtype = getinargx(lv->fn->type);
	if(inargtype != nil) {
		xoffset = 0;
		twobitwalktype1(inargtype, &xoffset, args);
	}
	*/
}

// Construct a disembodied instruction.
static Prog*
unlinkedprog(int as)
{
	Prog *p;

	p = mal(sizeof(*p));
	clearp(p);
	p->as = as;
	return p;
}

// Construct a new PCDATA instruction associated with and for the purposes of
// covering an existing instruction.
static Prog*
newpcdataprog(Prog *prog, int32 index)
{
	Node from;
	Node to;
	Prog *pcdata;

	nodconst(&from, types[TINT32], PCDATA_StackMapIndex);
	nodconst(&to, types[TINT32], index);
	pcdata = unlinkedprog(APCDATA);
	pcdata->lineno = prog->lineno;
	naddr(&from, &pcdata->from, 0);
	naddr(&to, &pcdata->to, 0);
	return pcdata;
}

// Returns true for instructions that are safe points that must be annotated
// with liveness information.
static int
issafepoint(Prog *prog)
{
	return prog->as == ATEXT || prog->as == ACALL;
}

// Initializes the sets for solving the live variables.  Visits all the
// instructions in each basic block to summarizes the information at each basic
// block
static void
livenessprologue(Liveness *lv)
{
	BasicBlock *bb;
	Bvec *uevar;
	Bvec *varkill;
	Prog *prog;
	int32 i;
	int32 nvars;

	nvars = arraylength(lv->vars);
	uevar = bvalloc(nvars);
	varkill = bvalloc(nvars);
	for(i = 0; i < arraylength(lv->cfg); i++) {
		bb = *(BasicBlock**)arrayget(lv->cfg, i);
		// Walk the block instructions backward and update the block
		// effects with the each prog effects.
		for(prog = bb->last; prog != nil; prog = prog->opt) {
			progeffects(prog, lv->vars, uevar, varkill);
			if(0) printeffects(prog, uevar, varkill);
			bvor(lv->varkill[i], lv->varkill[i], varkill);
			bvandnot(lv->uevar[i], lv->uevar[i], varkill);
			bvor(lv->uevar[i], lv->uevar[i], uevar);
		}
	}
	free(uevar);
	free(varkill);
}

// Solve the liveness dataflow equations.
static void
livenesssolve(Liveness *lv)
{
	BasicBlock *bb;
	BasicBlock *succ;
	Bvec *newlivein;
	Bvec *newliveout;
	int32 rpo;
	int32 i;
	int32 j;
	int change;

	// These temporary bitvectors exist to avoid successive allocations and
	// frees within the loop.
	newlivein = bvalloc(arraylength(lv->vars));
	newliveout = bvalloc(arraylength(lv->vars));

	// Iterate through the blocks in reverse round-robin fashion.  A work
	// queue might be slightly faster.  As is, the number of iterations is
	// so low that it hardly seems to be worth the complexity.
	change = 1;
	while(change != 0) {
		change = 0;
		// Walk blocks in the general direction of propagation.  This
		// improves convergence.
		for(i = arraylength(lv->cfg) - 1; i >= 0; i--) {
			// out[b] = \bigcup_{s \in succ[b]} in[s]
			bb = *(BasicBlock**)arrayget(lv->cfg, i);
			rpo = bb->rpo;
			bvresetall(newliveout);
			for(j = 0; j < arraylength(bb->succ); j++) {
				succ = *(BasicBlock**)arrayget(bb->succ, j);
				bvor(newliveout, newliveout, lv->livein[succ->rpo]);
			}
			if(bvcmp(lv->liveout[rpo], newliveout)) {
				change = 1;
				bvcopy(lv->liveout[rpo], newliveout);
			}
			// in[b] = uevar[b] \cup (out[b] \setminus varkill[b])
			bvandnot(newlivein, lv->liveout[rpo], lv->varkill[rpo]);
			bvor(lv->livein[rpo], newlivein, lv->uevar[rpo]);
		}
	}

	free(newlivein);
	free(newliveout);
}

// Visits all instructions in a basic block and computes a bit vector of live
// variables at each safe point locations.
static void
livenessepilogue(Liveness *lv)
{
	BasicBlock *bb;
	Bvec *livein;
	Bvec *liveout;
	Bvec *uevar;
	Bvec *varkill;
	Bvec *args;
	Bvec *locals;
	Prog *p;
	int32 i;
	int32 nvars;
	int32 pos;

	nvars = arraylength(lv->vars);
	livein = bvalloc(nvars);
	liveout = bvalloc(nvars);
	uevar = bvalloc(nvars);
	varkill = bvalloc(nvars);

	for(i = 0; i < arraylength(lv->cfg); i++) {
		bb = *(BasicBlock**)arrayget(lv->cfg, i);
		bvcopy(livein, lv->liveout[bb->rpo]);
		// Walk forward through the basic block instructions and
		// allocate and empty map for those instructions that need them
		for(p = bb->last; p != nil; p = p->opt) {
			if(!issafepoint(p))
				continue;

			// Allocate a bit vector for each class and facet of
			// value we are tracking.

			// Live stuff first.
			args = bvalloc(argswords() * BitsPerPointer);
			arrayadd(lv->argslivepointers, &args);
			locals = bvalloc(localswords() * BitsPerPointer);
			arrayadd(lv->livepointers, &locals);

			// Dead stuff second.
			args = bvalloc(argswords() * BitsPerPointer);
			arrayadd(lv->argsdeadvalues, &args);
			locals = bvalloc(localswords() * BitsPerPointer);
			arrayadd(lv->deadvalues, &locals);
		}

		// walk backward, emit pcdata and populate the maps
		pos = arraylength(lv->livepointers) - 1;
		if(pos < 0) {
			// the first block we encounter should have the ATEXT so
			// at no point should pos ever be less than zero.
			fatal("livenessepilogue");
		}

		for(p = bb->last; p != nil; p = p->opt) {
			// Propagate liveness information
			progeffects(p, lv->vars, uevar, varkill);
			bvcopy(liveout, livein);
			bvandnot(livein, liveout, varkill);
			bvor(livein, livein, uevar);
			if(printnoise){
				print("%P\n", p);
				printvars("uevar", uevar, lv->vars);
				printvars("varkill", varkill, lv->vars);
				printvars("livein", livein, lv->vars);
				printvars("liveout", liveout, lv->vars);
			}
			if(issafepoint(p)) {
				// Found an interesting instruction, record the
				// corresponding liveness information.  Only
				// CALL instructions need a PCDATA annotation.
				// The TEXT instruction annotation is implicit.
				if(p->as == ACALL) {
					if(isdeferreturn(p)) {
						// Because this runtime call
						// modifies its return address
						// to return back to itself,
						// emitting a PCDATA before the
						// call instruction will result
						// in an off by one error during
						// a stack walk.  Fortunately,
						// the compiler inserts a no-op
						// instruction before this call
						// so we can reliably anchor the
						// PCDATA to that instruction.
						splicebefore(lv, bb, newpcdataprog(p->opt, pos), p->opt);
					} else {
						splicebefore(lv, bb, newpcdataprog(p, pos), p);
					}
				}

				// Record live pointers.
				args = *(Bvec**)arrayget(lv->argslivepointers, pos);
				locals = *(Bvec**)arrayget(lv->livepointers, pos);
				twobitlivepointermap(lv, liveout, lv->vars, args, locals);

				// Record dead values.
				args = *(Bvec**)arrayget(lv->argsdeadvalues, pos);
				locals = *(Bvec**)arrayget(lv->deadvalues, pos);
				twobitdeadvaluemap(lv, liveout, lv->vars, args, locals);

				pos--;
			}
		}
	}

	free(livein);
	free(liveout);
	free(uevar);
	free(varkill);
}

// Dumps an array of bitmaps to a symbol as a sequence of uint32 values.  The
// first word dumped is the total number of bitmaps.  The second word is the
// length of the bitmaps.  All bitmaps are assumed to be of equal length.  The
// words that are followed are the raw bitmap words.  The arr argument is an
// array of Node*s.
static void
twobitwritesymbol(Array *arr, Sym *sym, Bvec *check)
{
	Bvec *bv;
	int off;
	uint32 bit;
	uint32 word;
	uint32 checkword;
	int32 i;
	int32 j;
	int32 len;
	int32 pos;

	len = arraylength(arr);
	// Dump the length of the bitmap array.
	off = duint32(sym, 0, len);
	for(i = 0; i < len; i++) {
		bv = *(Bvec**)arrayget(arr, i);
		// If we have been provided a check bitmap we can use it
		// to confirm that the bitmap we are dumping is a subset
		// of the check bitmap.
		if(check != nil) {
			for(j = 0; j < bv->n; j += 32) {
				word = bv->b[j/32];
				checkword = check->b[j/32];
				if(word != checkword) {
					// Found a mismatched word, find
					// the mismatched bit.
					for(pos = 0; pos < 32; pos++) {
						bit = 1 << pos;
						if((word & bit) && !(checkword & bit)) {
							print("twobitwritesymbol: expected %032b to be a subset of %032b\n", word, checkword);
							fatal("mismatch at bit position %d\n", pos);
						}
					}
				}
			}
		}
		// Dump the length of the bitmap.
		off = duint32(sym, off, bv->n);
		// Dump the words of the bitmap.
		for(j = 0; j < bv->n; j += 32) {
			word = bv->b[j/32];
			off = duint32(sym, off, word);
		}
	}
	ggloblsym(sym, off, 0, 1);
}

static void
printprog(Prog *p)
{
	while(p != nil) {
		print("%P\n", p);
		p = p->link;
	}
}

// Entry pointer for liveness analysis.  Constructs a complete CFG, solves for
// the liveness of pointer variables in the function, and emits a runtime data
// structure read by the garbage collector.
void
liveness(Node *fn, Prog *firstp, Sym *argssym, Sym *livesym, Sym *deadsym)
{
	Array *cfg;
	Array *vars;
	Liveness *lv;

	if(0) print("curfn->nname->sym->name is %s\n", curfn->nname->sym->name);
	if(0) printprog(firstp);
	checkptxt(fn, firstp);

	// Construct the global liveness state.
	cfg = newcfg(firstp);
	if(0) printcfg(cfg);
	vars = getvariables(fn);
	lv = newliveness(fn, firstp, cfg, vars);

	// Run the dataflow framework.
	livenessprologue(lv);
	if(0) livenessprintcfg(lv);
	livenesssolve(lv);
	if(0) livenessprintcfg(lv);
	livenessepilogue(lv);

	// Emit the live pointer map data structures
	twobitwritesymbol(lv->livepointers, livesym, nil);
	twobitwritesymbol(lv->argslivepointers, argssym, nil);

	// Optionally emit a dead value map data structure for locals.
	if(deadsym != nil)
		twobitwritesymbol(lv->deadvalues, deadsym, nil);

	// Free everything.
	freeliveness(lv);
	arrayfree(vars);
	freecfg(cfg);
}
