// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Garbage collector liveness bitmap generation.

// The command line flag -live causes this code to print debug information.
// The levels are:
//
//	-live (aka -live=1): print liveness lists as code warnings at safe points
//	-live=2: print an assembly listing with liveness annotations
//	-live=3: print information during each computation phase (much chattier)
//
// Each level includes the earlier output as well.

#include <u.h>
#include <libc.h>
#include "gg.h"
#include "opt.h"
#include "../ld/textflag.h"
#include "../../runtime/funcdata.h"
#include "../../runtime/mgc0.h"

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
	
	// For use during livenessepilogue.
	int lastbitmapindex;
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
	// The blocks are roughly in the same order as the instructions
	// in the function (first block has TEXT instruction, and so on).
	Array *cfg;

	// Summary sets of block effects.
	// The Bvec** is indexed by bb->rpo to yield a single Bvec*.
	// That bit vector is indexed by variable number (same as lv->vars).
	//
	// Computed during livenessprologue using only the content of
	// individual blocks:
	//
	//	uevar: upward exposed variables (used before set in block)
	//	varkill: killed variables (set in block)
	//	avarinit: addrtaken variables set or used (proof of initialization)
	//
	// Computed during livenesssolve using control flow information:
	//
	//	livein: variables live at block entry
	//	liveout: variables live at block exit
	//	avarinitany: addrtaken variables possibly initialized at block exit
	//		(initialized in block or at exit from any predecessor block)
	//	avarinitall: addrtaken variables certainly initialized at block exit
	//		(initialized in block or at exit from all predecessor blocks)
	Bvec **uevar;
	Bvec **varkill;
	Bvec **livein;
	Bvec **liveout;
	Bvec **avarinit;
	Bvec **avarinitany;
	Bvec **avarinitall;

	// An array with a bit vector for each safe point tracking live pointers
	// in the arguments and locals area, indexed by bb->rpo.
	Array *argslivepointers;
	Array *livepointers;
};

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

// Inserts prev before curr in the instruction
// stream.  Any control flow, such as branches or fall throughs, that target the
// existing instruction are adjusted to target the new instruction.
static void
splicebefore(Liveness *lv, BasicBlock *bb, Prog *prev, Prog *curr)
{
	Prog *next, tmp;

	USED(lv);

	// There may be other instructions pointing at curr,
	// and we want them to now point at prev. Instead of
	// trying to find all such instructions, swap the contents
	// so that the problem becomes inserting next after curr.
	// The "opt" field is the backward link in the linked list.

	// Overwrite curr's data with prev, but keep the list links.
	tmp = *curr;
	*curr = *prev;
	curr->opt = tmp.opt;
	curr->link = tmp.link;
	
	// Overwrite prev (now next) with curr's old data.
	next = prev;
	*next = tmp;
	next->opt = nil;
	next->link = nil;

	// Now insert next after curr.
	next->link = curr->link;
	next->opt = curr;
	curr->link = next;
	if(next->link && next->link->opt == curr)
		next->link->opt = next;

	if(bb->last == curr)
		bb->last = next;
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
// variables.
static Array*
getvariables(Node *fn)
{
	Array *result;
	NodeList *ll;

	result = arraynew(0, sizeof(Node*));
	for(ll = fn->dcl; ll != nil; ll = ll->next) {
		if(ll->n->op == ONAME) {
			// In order for GODEBUG=gcdead=1 to work, each bitmap needs
			// to contain information about all variables covered by the bitmap.
			// For local variables, the bitmap only covers the stkptrsize
			// bytes in the frame where variables containing pointers live.
			// For arguments and results, the bitmap covers all variables,
			// so we must include all the variables, even the ones without
			// pointers.
			//
			// The Node.opt field is available for use by optimization passes.
			// We use it to hold the index of the node in the variables array, plus 1
			// (so that 0 means the Node is not in the variables array).
			// Each pass should clear opt when done, but you never know,
			// so clear them all ourselves too.
			// The Node.curfn field is supposed to be set to the current function
			// already, but for some compiler-introduced names it seems not to be,
			// so fix that here.
			// Later, when we want to find the index of a node in the variables list,
			// we will check that n->curfn == curfn and n->opt > 0. Then n->opt - 1
			// is the index in the variables list.
			ll->n->opt = nil;
			ll->n->curfn = curfn;
			switch(ll->n->class) {
			case PAUTO:
				if(haspointers(ll->n->type)) {
					ll->n->opt = (void*)(uintptr)(arraylength(result)+1);
					arrayadd(result, &ll->n);
				}
				break;
			case PPARAM:
			case PPARAMOUT:
				ll->n->opt = (void*)(uintptr)(arraylength(result)+1);
				arrayadd(result, &ll->n);
				break;
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

			// Stop before an unreachable RET, to avoid creating
			// unreachable control flow nodes.
			if(p->link != nil && p->link->as == ARET && p->link->mode == 1)
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
			case AUNDEF:
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
	if(bb->rpo == -1) {
		print("newcfg: unreachable basic block for %P\n", bb->last);
		printcfg(cfg);
		fatal("newcfg: invalid control flow graph");
	}

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
	char *names[] = { ".fp", ".args", nil };
	int i;

	if(node->sym != nil && node->sym->name != nil)
		for(i = 0; names[i] != nil; i++)
			if(strcmp(node->sym->name, names[i]) == 0)
				return 1;
	return 0;
}

// Computes the effects of an instruction on a set of
// variables.  The vars argument is an array of Node*s.
//
// The output vectors give bits for variables:
//	uevar - used by this instruction
//	varkill - killed by this instruction
//		for variables without address taken, means variable was set
//		for variables with address taken, means variable was marked dead
//	avarinit - initialized or referred to by this instruction,
//		only for variables with address taken but not escaping to heap
//
// The avarinit output serves as a signal that the data has been
// initialized, because any use of a variable must come after its
// initialization.
static void
progeffects(Prog *prog, Array *vars, Bvec *uevar, Bvec *varkill, Bvec *avarinit)
{
	ProgInfo info;
	Addr *from;
	Addr *to;
	Node *node;
	int32 i;
	int32 pos;

	bvresetall(uevar);
	bvresetall(varkill);
	bvresetall(avarinit);

	proginfo(&info, prog);
	if(prog->as == ARET) {
		// Return instructions implicitly read all the arguments.  For
		// the sake of correctness, out arguments must be read.  For the
		// sake of backtrace quality, we read in arguments as well.
		//
		// A return instruction with a p->to is a tail return, which brings
		// the stack pointer back up (if it ever went down) and then jumps
		// to a new function entirely. That form of instruction must read
		// all the parameters for correctness, and similarly it must not
		// read the out arguments - they won't be set until the new
		// function runs.
		for(i = 0; i < arraylength(vars); i++) {
			node = *(Node**)arrayget(vars, i);
			switch(node->class & ~PHEAP) {
			case PPARAM:
				bvset(uevar, i);
				break;
			case PPARAMOUT:
				// If the result had its address taken, it is being tracked
				// by the avarinit code, which does not use uevar.
				// If we added it to uevar too, we'd not see any kill
				// and decide that the varible was live entry, which it is not.
				// So only use uevar in the non-addrtaken case.
				// The p->to.type == D_NONE limits the bvset to
				// non-tail-call return instructions; see note above
				// the for loop for details.
				if(!node->addrtaken && prog->to.type == D_NONE)
					bvset(uevar, i);
				break;
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
				if(node->addrtaken)
					bvset(avarinit, i);
				bvset(varkill, i);
				break;
			}
		}
		return;
	}
	if(info.flags & (LeftRead | LeftWrite | LeftAddr)) {
		from = &prog->from;
		if (from->node != nil && from->sym != nil && from->node->curfn == curfn) {
			switch(from->node->class & ~PHEAP) {
			case PAUTO:
			case PPARAM:
			case PPARAMOUT:
				pos = (int)(uintptr)from->node->opt - 1; // index in vars
				if(pos == -1)
					goto Next;
				if(pos >= arraylength(vars) || *(Node**)arrayget(vars, pos) != from->node)
					fatal("bad bookkeeping in liveness %N %d", from->node, pos);
				if(from->node->addrtaken) {
					bvset(avarinit, pos);
				} else {
					if(info.flags & (LeftRead | LeftAddr))
						bvset(uevar, pos);
					if(info.flags & LeftWrite)
						if(from->node != nil && !isfat(from->node->type))
							bvset(varkill, pos);
				}
			}
		}
	}
Next:
	if(info.flags & (RightRead | RightWrite | RightAddr)) {
		to = &prog->to;
		if (to->node != nil && to->sym != nil && to->node->curfn == curfn) {
			switch(to->node->class & ~PHEAP) {
			case PAUTO:
			case PPARAM:
			case PPARAMOUT:
				pos = (int)(uintptr)to->node->opt - 1; // index in vars
				if(pos == -1)
					goto Next1;
				if(pos >= arraylength(vars) || *(Node**)arrayget(vars, pos) != to->node)
					fatal("bad bookkeeping in liveness %N %d", to->node, pos);
				if(to->node->addrtaken) {
					if(prog->as != AVARKILL)
						bvset(avarinit, pos);
					if(prog->as == AVARDEF || prog->as == AVARKILL)
						bvset(varkill, pos);
				} else {
					// RightRead is a read, obviously.
					// RightAddr by itself is also implicitly a read.
					//
					// RightAddr|RightWrite means that the address is being taken
					// but only so that the instruction can write to the value.
					// It is not a read. It is equivalent to RightWrite except that
					// having the RightAddr bit set keeps the registerizer from
					// trying to substitute a register for the memory location.
					if((info.flags & RightRead) || (info.flags & (RightAddr|RightWrite)) == RightAddr)
						bvset(uevar, pos);
					if(info.flags & RightWrite)
						if(to->node != nil && (!isfat(to->node->type) || prog->as == AVARDEF))
							bvset(varkill, pos);
				}
			}
		}
	}
Next1:;
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
	result->avarinit = xmalloc(sizeof(Bvec*) * nblocks);
	result->avarinitany = xmalloc(sizeof(Bvec*) * nblocks);
	result->avarinitall = xmalloc(sizeof(Bvec*) * nblocks);

	nvars = arraylength(vars);
	for(i = 0; i < nblocks; i++) {
		result->uevar[i] = bvalloc(nvars);
		result->varkill[i] = bvalloc(nvars);
		result->livein[i] = bvalloc(nvars);
		result->liveout[i] = bvalloc(nvars);
		result->avarinit[i] = bvalloc(nvars);
		result->avarinitany[i] = bvalloc(nvars);
		result->avarinitall[i] = bvalloc(nvars);
	}

	result->livepointers = arraynew(0, sizeof(Bvec*));
	result->argslivepointers = arraynew(0, sizeof(Bvec*));
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

	for(i = 0; i < arraylength(lv->cfg); i++) {
		free(lv->uevar[i]);
		free(lv->varkill[i]);
		free(lv->livein[i]);
		free(lv->liveout[i]);
		free(lv->avarinit[i]);
		free(lv->avarinitany[i]);
		free(lv->avarinitall[i]);
	}

	free(lv->uevar);
	free(lv->varkill);
	free(lv->livein);
	free(lv->liveout);
	free(lv->avarinit);
	free(lv->avarinitany);
	free(lv->avarinitall);

	free(lv);
}

static void
printeffects(Prog *p, Bvec *uevar, Bvec *varkill, Bvec *avarinit)
{
	print("effects of %P", p);
	print("\nuevar: ");
	bvprint(uevar);
	print("\nvarkill: ");
	bvprint(varkill);
	print("\navarinit: ");
	bvprint(avarinit);
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
	printvars("\tavarinit", lv->avarinit[bb->rpo], lv->vars);
	printvars("\tavarinitany", lv->avarinitany[bb->rpo], lv->vars);
	printvars("\tavarinitall", lv->avarinitall[bb->rpo], lv->vars);

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
checkauto(Node *fn, Prog *p, Node *n)
{
	NodeList *l;

	for(l = fn->dcl; l != nil; l = l->next)
		if(l->n->op == ONAME && l->n->class == PAUTO && l->n == n)
			return;

	print("checkauto %N: %N (%p; class=%d) not found in %P\n", curfn, n, n, n->class, p);
	for(l = fn->dcl; l != nil; l = l->next)
		print("\t%N (%p; class=%d)\n", l->n, l->n, l->n->class);
	yyerror("checkauto: invariant lost");
}

static void
checkparam(Node *fn, Prog *p, Node *n)
{
	NodeList *l;
	Node *a;
	int class;

	if(isfunny(n))
		return;
	for(l = fn->dcl; l != nil; l = l->next) {
		a = l->n;
		class = a->class & ~PHEAP;
		if(a->op == ONAME && (class == PPARAM || class == PPARAMOUT) && a == n)
			return;
	}

	print("checkparam %N: %N (%p; class=%d) not found in %P\n", curfn, n, n, n->class, p);
	for(l = fn->dcl; l != nil; l = l->next)
		print("\t%N (%p; class=%d)\n", l->n, l->n, l->n->class);
	yyerror("checkparam: invariant lost");
}

static void
checkprog(Node *fn, Prog *p)
{
	if(p->from.type == D_AUTO)
		checkauto(fn, p, p->from.node);
	if(p->from.type == D_PARAM)
		checkparam(fn, p, p->from.node);
	if(p->to.type == D_AUTO)
		checkauto(fn, p, p->to.node);
	if(p->to.type == D_PARAM)
		checkparam(fn, p, p->to.node);
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

	if(debuglive == 0)
		return;

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

// NOTE: The bitmap for a specific type t should be cached in t after the first run
// and then simply copied into bv at the correct offset on future calls with
// the same type t. On https://rsc.googlecode.com/hg/testdata/slow.go, twobitwalktype1
// accounts for 40% of the 6g execution time.
void
twobitwalktype1(Type *t, vlong *xoffset, Bvec *bv)
{
	vlong fieldoffset;
	vlong i;
	vlong o;
	Type *t1;

	if(t->align > 0 && (*xoffset & (t->align - 1)) != 0)
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
		for(i = 0; i < t->width; i++) {
			bvset(bv, ((*xoffset + i) / widthptr) * BitsPerPointer); // 1 = live scalar
		}
		*xoffset += t->width;
		break;

	case TPTR32:
	case TPTR64:
	case TUNSAFEPTR:
	case TFUNC:
	case TCHAN:
	case TMAP:
		if((*xoffset & (widthptr-1)) != 0)
			fatal("twobitwalktype1: invalid alignment, %T", t);
		bvset(bv, (*xoffset / widthptr) * BitsPerPointer + 1); // 2 = live ptr
		*xoffset += t->width;
		break;

	case TSTRING:
		// struct { byte *str; intgo len; }
		if((*xoffset & (widthptr-1)) != 0)
			fatal("twobitwalktype1: invalid alignment, %T", t);
		bvset(bv, (*xoffset / widthptr) * BitsPerPointer + 1); // 2 = live ptr in first slot
		*xoffset += t->width;
		break;

	case TINTER:
		// struct { Itab *tab;	union { void *ptr, uintptr val } data; }
		// or, when isnilinter(t)==true:
		// struct { Type *type; union { void *ptr, uintptr val } data; }
		if((*xoffset & (widthptr-1)) != 0)
			fatal("twobitwalktype1: invalid alignment, %T", t);
		bvset(bv, ((*xoffset / widthptr) * BitsPerPointer) + 0);
		bvset(bv, ((*xoffset / widthptr) * BitsPerPointer) + 1); // 3 = multiword
		// next word contains 2 = Iface, 3 = Eface
		if(isnilinter(t)) {
			bvset(bv, ((*xoffset / widthptr) * BitsPerPointer) + 2);
			bvset(bv, ((*xoffset / widthptr) * BitsPerPointer) + 3);
		} else {
			bvset(bv, ((*xoffset / widthptr) * BitsPerPointer) + 3);
		}
		*xoffset += t->width;
		break;

	case TARRAY:
		// The value of t->bound is -1 for slices types and >0 for
		// for fixed array types.  All other values are invalid.
		if(t->bound < -1)
			fatal("twobitwalktype1: invalid bound, %T", t);
		if(isslice(t)) {
			// struct { byte *array; uintgo len; uintgo cap; }
			if((*xoffset & (widthptr-1)) != 0)
				fatal("twobitwalktype1: invalid TARRAY alignment, %T", t);
			bvset(bv, (*xoffset / widthptr) * BitsPerPointer + 1); // 2 = live ptr in first slot
			*xoffset += t->width;
		} else
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

	for(i = 0; (i = bvnext(liveout, i)) >= 0; i++) {
		node = *(Node**)arrayget(vars, i);
		switch(node->class) {
		case PAUTO:
			xoffset = node->xoffset + stkptrsize;
			twobitwalktype1(node->type, &xoffset, locals);
			break;
		case PPARAM:
		case PPARAMOUT:
			xoffset = node->xoffset;
			twobitwalktype1(node->type, &xoffset, args);
			break;
		}
	}
	
	// The node list only contains declared names.
	// If the receiver or arguments are unnamed, they will be omitted
	// from the list above. Preserve those values - even though they are unused -
	// in order to keep their addresses live for use in stack traces.
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
	Node from, to;
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
	Bvec *uevar, *varkill, *avarinit;
	Prog *p;
	int32 i;
	int32 nvars;

	nvars = arraylength(lv->vars);
	uevar = bvalloc(nvars);
	varkill = bvalloc(nvars);
	avarinit = bvalloc(nvars);
	for(i = 0; i < arraylength(lv->cfg); i++) {
		bb = *(BasicBlock**)arrayget(lv->cfg, i);
		// Walk the block instructions backward and update the block
		// effects with the each prog effects.
		for(p = bb->last; p != nil; p = p->opt) {
			progeffects(p, lv->vars, uevar, varkill, avarinit);
			if(debuglive >= 3)
				printeffects(p, uevar, varkill, avarinit);
			bvor(lv->varkill[i], lv->varkill[i], varkill);
			bvandnot(lv->uevar[i], lv->uevar[i], varkill);
			bvor(lv->uevar[i], lv->uevar[i], uevar);			
		}
		// Walk the block instructions forward to update avarinit bits.
		// avarinit describes the effect at the end of the block, not the beginning.
		bvresetall(varkill);
		for(p = bb->first;; p = p->link) {
			progeffects(p, lv->vars, uevar, varkill, avarinit);
			if(debuglive >= 3)
				printeffects(p, uevar, varkill, avarinit);
			bvandnot(lv->avarinit[i], lv->avarinit[i], varkill);
			bvor(lv->avarinit[i], lv->avarinit[i], avarinit);
			if(p == bb->last)
				break;
		}
	}
	free(uevar);
	free(varkill);
	free(avarinit);
}

// Solve the liveness dataflow equations.
static void
livenesssolve(Liveness *lv)
{
	BasicBlock *bb, *succ, *pred;
	Bvec *newlivein, *newliveout, *any, *all;
	int32 rpo, i, j, change;

	// These temporary bitvectors exist to avoid successive allocations and
	// frees within the loop.
	newlivein = bvalloc(arraylength(lv->vars));
	newliveout = bvalloc(arraylength(lv->vars));
	any = bvalloc(arraylength(lv->vars));
	all = bvalloc(arraylength(lv->vars));

	// Push avarinitall, avarinitany forward.
	// avarinitall says the addressed var is initialized along all paths reaching the block exit.
	// avarinitany says the addressed var is initialized along some path reaching the block exit.
	for(i = 0; i < arraylength(lv->cfg); i++) {
		bb = *(BasicBlock**)arrayget(lv->cfg, i);
		rpo = bb->rpo;
		if(i == 0)
			bvcopy(lv->avarinitall[rpo], lv->avarinit[rpo]);
		else {
			bvresetall(lv->avarinitall[rpo]);
			bvnot(lv->avarinitall[rpo]);
		}
		bvcopy(lv->avarinitany[rpo], lv->avarinit[rpo]);
	}

	change = 1;
	while(change != 0) {
		change = 0;
		for(i = 0; i < arraylength(lv->cfg); i++) {
			bb = *(BasicBlock**)arrayget(lv->cfg, i);
			rpo = bb->rpo;
			bvresetall(any);
			bvresetall(all);
			for(j = 0; j < arraylength(bb->pred); j++) {
				pred = *(BasicBlock**)arrayget(bb->pred, j);
				if(j == 0) {
					bvcopy(any, lv->avarinitany[pred->rpo]);
					bvcopy(all, lv->avarinitall[pred->rpo]);
				} else {
					bvor(any, any, lv->avarinitany[pred->rpo]);
					bvand(all, all, lv->avarinitall[pred->rpo]);
				}
			}
			bvandnot(any, any, lv->varkill[rpo]);
			bvandnot(all, all, lv->varkill[rpo]);
			bvor(any, any, lv->avarinit[rpo]);
			bvor(all, all, lv->avarinit[rpo]);
			if(bvcmp(any, lv->avarinitany[rpo])) {
				change = 1;
				bvcopy(lv->avarinitany[rpo], any);
			}
			if(bvcmp(all, lv->avarinitall[rpo])) {
				change = 1;
				bvcopy(lv->avarinitall[rpo], all);
			}
		}
	}

	// Iterate through the blocks in reverse round-robin fashion.  A work
	// queue might be slightly faster.  As is, the number of iterations is
	// so low that it hardly seems to be worth the complexity.
	change = 1;
	while(change != 0) {
		change = 0;
		// Walk blocks in the general direction of propagation.  This
		// improves convergence.
		for(i = arraylength(lv->cfg) - 1; i >= 0; i--) {
			// A variable is live on output from this block
			// if it is live on input to some successor.
			//
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

			// A variable is live on input to this block
			// if it is live on output from this block and
			// not set by the code in this block.
			//
			// in[b] = uevar[b] \cup (out[b] \setminus varkill[b])
			bvandnot(newlivein, lv->liveout[rpo], lv->varkill[rpo]);
			bvor(lv->livein[rpo], newlivein, lv->uevar[rpo]);
		}
	}

	free(newlivein);
	free(newliveout);
	free(any);
	free(all);
}

// This function is slow but it is only used for generating debug prints.
// Check whether n is marked live in args/locals.
static int
islive(Node *n, Bvec *args, Bvec *locals)
{
	int i;

	switch(n->class) {
	case PPARAM:
	case PPARAMOUT:
		for(i = 0; i < n->type->width/widthptr*BitsPerPointer; i++)
			if(bvget(args, n->xoffset/widthptr*BitsPerPointer + i))
				return 1;
		break;
	case PAUTO:
		for(i = 0; i < n->type->width/widthptr*BitsPerPointer; i++)
			if(bvget(locals, (n->xoffset + stkptrsize)/widthptr*BitsPerPointer + i))
				return 1;
		break;
	}
	return 0;
}

// Visits all instructions in a basic block and computes a bit vector of live
// variables at each safe point locations.
static void
livenessepilogue(Liveness *lv)
{
	BasicBlock *bb, *pred;
	Bvec *ambig, *livein, *liveout, *uevar, *varkill, *args, *locals, *avarinit, *any, *all;
	Node *n;
	Prog *p, *next;
	int32 i, j, numlive, startmsg, nmsg, nvars, pos;
	vlong xoffset;
	char **msg;
	Fmt fmt;

	nvars = arraylength(lv->vars);
	livein = bvalloc(nvars);
	liveout = bvalloc(nvars);
	uevar = bvalloc(nvars);
	varkill = bvalloc(nvars);
	avarinit = bvalloc(nvars);
	any = bvalloc(nvars);
	all = bvalloc(nvars);
	ambig = bvalloc(localswords() * BitsPerPointer);
	msg = nil;
	nmsg = 0;
	startmsg = 0;

	for(i = 0; i < arraylength(lv->cfg); i++) {
		bb = *(BasicBlock**)arrayget(lv->cfg, i);
		
		// Compute avarinitany and avarinitall for entry to block.
		// This duplicates information known during livenesssolve
		// but avoids storing two more vectors for each block.
		bvresetall(any);
		bvresetall(all);
		for(j = 0; j < arraylength(bb->pred); j++) {
			pred = *(BasicBlock**)arrayget(bb->pred, j);
			if(j == 0) {
				bvcopy(any, lv->avarinitany[pred->rpo]);
				bvcopy(all, lv->avarinitall[pred->rpo]);
			} else {
				bvor(any, any, lv->avarinitany[pred->rpo]);
				bvand(all, all, lv->avarinitall[pred->rpo]);
			}
		}

		// Walk forward through the basic block instructions and
		// allocate liveness maps for those instructions that need them.
		// Seed the maps with information about the addrtaken variables.
		for(p = bb->first;; p = p->link) {
			progeffects(p, lv->vars, uevar, varkill, avarinit);
			bvandnot(any, any, varkill);
			bvandnot(all, all, varkill);
			bvor(any, any, avarinit);
			bvor(all, all, avarinit);

			if(issafepoint(p)) {
				// Annotate ambiguously live variables so that they can
				// be zeroed at function entry.
				// livein and liveout are dead here and used as temporaries.
				// For now, only enabled when using GOEXPERIMENT=precisestack
				// during make.bash / all.bash.
				if(precisestack_enabled) {
					bvresetall(livein);
					bvandnot(liveout, any, all);
					if(!bvisempty(liveout)) {
						for(pos = 0; pos < liveout->n; pos++) {
							if(!bvget(liveout, pos))
								continue;
							bvset(all, pos); // silence future warnings in this block
							n = *(Node**)arrayget(lv->vars, pos);
							if(!n->needzero) {
								n->needzero = 1;
								if(debuglive >= 1)
									warnl(p->lineno, "%N: %lN is ambiguously live", curfn->nname, n);
								// Record in 'ambiguous' bitmap.
								xoffset = n->xoffset + stkptrsize;
								twobitwalktype1(n->type, &xoffset, ambig);
							}
						}
					}
				}
	
				// Allocate a bit vector for each class and facet of
				// value we are tracking.
	
				// Live stuff first.
				args = bvalloc(argswords() * BitsPerPointer);
				arrayadd(lv->argslivepointers, &args);
				locals = bvalloc(localswords() * BitsPerPointer);
				arrayadd(lv->livepointers, &locals);

				if(debuglive >= 3) {
					print("%P\n", p);
					printvars("avarinitany", any, lv->vars);
				}

				// Record any values with an "address taken" reaching
				// this code position as live. Must do now instead of below
				// because the any/all calculation requires walking forward
				// over the block (as this loop does), while the liveout
				// requires walking backward (as the next loop does).
				twobitlivepointermap(lv, any, lv->vars, args, locals);
			}
			
			if(p == bb->last)
				break;
		}
		bb->lastbitmapindex = arraylength(lv->livepointers) - 1;
	}
	
	for(i = 0; i < arraylength(lv->cfg); i++) {
		bb = *(BasicBlock**)arrayget(lv->cfg, i);
		
		if(debuglive >= 1 && strcmp(curfn->nname->sym->name, "init") != 0 && curfn->nname->sym->name[0] != '.') {
			nmsg = arraylength(lv->livepointers);
			startmsg = nmsg;
			msg = xmalloc(nmsg*sizeof msg[0]);
			for(j=0; j<nmsg; j++)
				msg[j] = nil;
		}

		// walk backward, emit pcdata and populate the maps
		pos = bb->lastbitmapindex;
		if(pos < 0) {
			// the first block we encounter should have the ATEXT so
			// at no point should pos ever be less than zero.
			fatal("livenessepilogue");
		}

		bvcopy(livein, lv->liveout[bb->rpo]);
		for(p = bb->last; p != nil; p = next) {
			next = p->opt; // splicebefore modifies p->opt
			// Propagate liveness information
			progeffects(p, lv->vars, uevar, varkill, avarinit);
			bvcopy(liveout, livein);
			bvandnot(livein, liveout, varkill);
			bvor(livein, livein, uevar);
			if(debuglive >= 3 && issafepoint(p)){
				print("%P\n", p);
				printvars("uevar", uevar, lv->vars);
				printvars("varkill", varkill, lv->vars);
				printvars("livein", livein, lv->vars);
				printvars("liveout", liveout, lv->vars);
			}
			if(issafepoint(p)) {
				// Found an interesting instruction, record the
				// corresponding liveness information.  
				
				// Useful sanity check: on entry to the function,
				// the only things that can possibly be live are the
				// input parameters.
				if(p->as == ATEXT) {
					for(j = 0; j < liveout->n; j++) {
						if(!bvget(liveout, j))
							continue;
						n = *(Node**)arrayget(lv->vars, j);
						if(n->class != PPARAM)
							yyerrorl(p->lineno, "internal error: %N %lN recorded as live on entry", curfn->nname, n);
					}
				}

				// Record live pointers.
				args = *(Bvec**)arrayget(lv->argslivepointers, pos);
				locals = *(Bvec**)arrayget(lv->livepointers, pos);
				twobitlivepointermap(lv, liveout, lv->vars, args, locals);
				
				// Ambiguously live variables are zeroed immediately after
				// function entry. Mark them live for all the non-entry bitmaps
				// so that GODEBUG=gcdead=1 mode does not poison them.
				if(p->as == ACALL)
					bvor(locals, locals, ambig);

				// Show live pointer bitmaps.
				// We're interpreting the args and locals bitmap instead of liveout so that we
				// include the bits added by the avarinit logic in the
				// previous loop.
				if(msg != nil) {
					fmtstrinit(&fmt);
					fmtprint(&fmt, "%L: live at ", p->lineno);
					if(p->as == ACALL && p->to.node)
						fmtprint(&fmt, "call to %s:", p->to.node->sym->name);
					else if(p->as == ACALL)
						fmtprint(&fmt, "indirect call:");
					else
						fmtprint(&fmt, "entry to %s:", p->from.node->sym->name);
					numlive = 0;
					for(j = 0; j < arraylength(lv->vars); j++) {
						n = *(Node**)arrayget(lv->vars, j);
						if(islive(n, args, locals)) {
							fmtprint(&fmt, " %N", n);
							numlive++;
						}
					}
					fmtprint(&fmt, "\n");
					if(numlive == 0) // squelch message
						free(fmtstrflush(&fmt));
					else
						msg[--startmsg] = fmtstrflush(&fmt);
				}

				// Only CALL instructions need a PCDATA annotation.
				// The TEXT instruction annotation is implicit.
				if(p->as == ACALL) {
					if(isdeferreturn(p)) {
						// runtime.deferreturn modifies its return address to return
						// back to the CALL, not to the subsequent instruction.
						// Because the return comes back one instruction early,
						// the PCDATA must begin one instruction early too.
						// The instruction before a call to deferreturn is always a
						// no-op, to keep PC-specific data unambiguous.
						splicebefore(lv, bb, newpcdataprog(p->opt, pos), p->opt);
					} else {
						splicebefore(lv, bb, newpcdataprog(p, pos), p);
					}
				}

				pos--;
			}
		}
		if(msg != nil) {
			for(j=startmsg; j<nmsg; j++) 
				if(msg[j] != nil)
					print("%s", msg[j]);
			free(msg);
			msg = nil;
			nmsg = 0;
			startmsg = 0;
		}
	}

	free(livein);
	free(liveout);
	free(uevar);
	free(varkill);
	free(avarinit);
	free(any);
	free(all);
	free(ambig);
	
	flusherrors();
}

// FNV-1 hash function constants.
#define H0 2166136261UL
#define Hp 16777619UL
/*c2go
enum
{
	H0 = 2166136261,
	Hp = 16777619,
};
*/

static uint32
hashbitmap(uint32 h, Bvec *bv)
{
	uchar *p, *ep;
	
	p = (uchar*)bv->b;
	ep = p + 4*((bv->n+31)/32);
	while(p < ep)
		h = (h*Hp) ^ *p++;
	return h;
}

// Compact liveness information by coalescing identical per-call-site bitmaps.
// The merging only happens for a single function, not across the entire binary.
//
// There are actually two lists of bitmaps, one list for the local variables and one
// list for the function arguments. Both lists are indexed by the same PCDATA
// index, so the corresponding pairs must be considered together when
// merging duplicates. The argument bitmaps change much less often during
// function execution than the local variable bitmaps, so it is possible that
// we could introduce a separate PCDATA index for arguments vs locals and
// then compact the set of argument bitmaps separately from the set of
// local variable bitmaps. As of 2014-04-02, doing this to the godoc binary
// is actually a net loss: we save about 50k of argument bitmaps but the new
// PCDATA tables cost about 100k. So for now we keep using a single index for
// both bitmap lists.
static void
livenesscompact(Liveness *lv)
{
	int *table, *remap, i, j, n, tablesize, uniq;
	uint32 h;
	Bvec *local, *arg, *jlocal, *jarg;
	Prog *p;

	// Linear probing hash table of bitmaps seen so far.
	// The hash table has 4n entries to keep the linear
	// scan short. An entry of -1 indicates an empty slot.
	n = arraylength(lv->livepointers);
	tablesize = 4*n;
	table = xmalloc(tablesize*sizeof table[0]);
	memset(table, 0xff, tablesize*sizeof table[0]);
	
	// remap[i] = the new index of the old bit vector #i.
	remap = xmalloc(n*sizeof remap[0]);
	memset(remap, 0xff, n*sizeof remap[0]);
	uniq = 0; // unique tables found so far

	// Consider bit vectors in turn.
	// If new, assign next number using uniq,
	// record in remap, record in lv->livepointers and lv->argslivepointers
	// under the new index, and add entry to hash table.
	// If already seen, record earlier index in remap and free bitmaps.
	for(i=0; i<n; i++) {
		local = *(Bvec**)arrayget(lv->livepointers, i);
		arg = *(Bvec**)arrayget(lv->argslivepointers, i);
		h = hashbitmap(hashbitmap(H0, local), arg) % tablesize;

		for(;;) {
			j = table[h];
			if(j < 0)
				break;
			jlocal = *(Bvec**)arrayget(lv->livepointers, j);
			jarg = *(Bvec**)arrayget(lv->argslivepointers, j);
			if(bvcmp(local, jlocal) == 0 && bvcmp(arg, jarg) == 0) {
				free(local);
				free(arg);
				remap[i] = j;
				goto Next;
			}
			if(++h == tablesize)
				h = 0;
		}
		table[h] = uniq;
		remap[i] = uniq;
		*(Bvec**)arrayget(lv->livepointers, uniq) = local;
		*(Bvec**)arrayget(lv->argslivepointers, uniq) = arg;
		uniq++;
	Next:;
	}

	// We've already reordered lv->livepointers[0:uniq]
	// and lv->argslivepointers[0:uniq] and freed the bitmaps
	// we don't need anymore. Clear the pointers later in the
	// array so that we can tell where the coalesced bitmaps stop
	// and so that we don't double-free when cleaning up.
	for(j=uniq; j<n; j++) {
		*(Bvec**)arrayget(lv->livepointers, j) = nil;
		*(Bvec**)arrayget(lv->argslivepointers, j) = nil;
	}
	
	// Rewrite PCDATA instructions to use new numbering.
	for(p=lv->ptxt; p != P; p=p->link) {
		if(p->as == APCDATA && p->from.offset == PCDATA_StackMapIndex) {
			i = p->to.offset;
			if(i >= 0)
				p->to.offset = remap[i];
		}
	}

	free(table);
	free(remap);
}

static int
printbitset(int printed, char *name, Array *vars, Bvec *bits)
{
	int i, started;
	Node *n;

	started = 0;	
	for(i=0; i<arraylength(vars); i++) {
		if(!bvget(bits, i))
			continue;
		if(!started) {
			if(!printed)
				print("\t");
			else
				print(" ");
			started = 1;
			printed = 1;
			print("%s=", name);
		} else {
			print(",");
		}
		n = *(Node**)arrayget(vars, i);
		print("%s", n->sym->name);
	}
	return printed;
}

// Prints the computed liveness information and inputs, for debugging.
// This format synthesizes the information used during the multiple passes
// into a single presentation.
static void
livenessprintdebug(Liveness *lv)
{
	int i, j, pcdata, printed;
	BasicBlock *bb;
	Prog *p;
	Bvec *uevar, *varkill, *avarinit, *args, *locals;
	Node *n;

	print("liveness: %s\n", curfn->nname->sym->name);

	uevar = bvalloc(arraylength(lv->vars));
	varkill = bvalloc(arraylength(lv->vars));
	avarinit = bvalloc(arraylength(lv->vars));

	pcdata = 0;
	for(i = 0; i < arraylength(lv->cfg); i++) {
		if(i > 0)
			print("\n");
		bb = *(BasicBlock**)arrayget(lv->cfg, i);

		// bb#0 pred=1,2 succ=3,4
		print("bb#%d pred=", i);
		for(j = 0; j < arraylength(bb->pred); j++) {
			if(j > 0)
				print(",");
			print("%d", (*(BasicBlock**)arrayget(bb->pred, j))->rpo);
		}
		print(" succ=");
		for(j = 0; j < arraylength(bb->succ); j++) {
			if(j > 0)
				print(",");
			print("%d", (*(BasicBlock**)arrayget(bb->succ, j))->rpo);
		}
		print("\n");
		
		// initial settings
		printed = 0;
		printed = printbitset(printed, "uevar", lv->vars, lv->uevar[bb->rpo]);
		printed = printbitset(printed, "livein", lv->vars, lv->livein[bb->rpo]);
		if(printed)
			print("\n");
		
		// program listing, with individual effects listed
		for(p = bb->first;; p = p->link) {
			print("%P\n", p);
			if(p->as == APCDATA && p->from.offset == PCDATA_StackMapIndex)
				pcdata = p->to.offset;
			progeffects(p, lv->vars, uevar, varkill, avarinit);
			printed = 0;
			printed = printbitset(printed, "uevar", lv->vars, uevar);
			printed = printbitset(printed, "varkill", lv->vars, varkill);
			printed = printbitset(printed, "avarinit", lv->vars, avarinit);
			if(printed)
				print("\n");
			if(issafepoint(p)) {
				args = *(Bvec**)arrayget(lv->argslivepointers, pcdata);
				locals = *(Bvec**)arrayget(lv->livepointers, pcdata);
				print("\tlive=");
				printed = 0;
				for(j = 0; j < arraylength(lv->vars); j++) {
					n = *(Node**)arrayget(lv->vars, j);
					if(islive(n, args, locals)) {
						if(printed++)
							print(",");
						print("%N", n);
					}
				}
				print("\n");
			}
			if(p == bb->last)
				break;
		}
		
		// bb bitsets
		print("end\n");
		printed = printbitset(printed, "varkill", lv->vars, lv->varkill[bb->rpo]);
		printed = printbitset(printed, "liveout", lv->vars, lv->liveout[bb->rpo]);
		printed = printbitset(printed, "avarinit", lv->vars, lv->avarinit[bb->rpo]);
		printed = printbitset(printed, "avarinitany", lv->vars, lv->avarinitany[bb->rpo]);
		printed = printbitset(printed, "avarinitall", lv->vars, lv->avarinitall[bb->rpo]);
		if(printed)
			print("\n");
	}
	print("\n");

	free(uevar);
	free(varkill);
	free(avarinit);
}

// Dumps an array of bitmaps to a symbol as a sequence of uint32 values.  The
// first word dumped is the total number of bitmaps.  The second word is the
// length of the bitmaps.  All bitmaps are assumed to be of equal length.  The
// words that are followed are the raw bitmap words.  The arr argument is an
// array of Node*s.
static void
twobitwritesymbol(Array *arr, Sym *sym)
{
	Bvec *bv;
	int off, i, j, len;
	uint32 word;

	len = arraylength(arr);
	off = 0;
	off += 4; // number of bitmaps, to fill in later
	bv = *(Bvec**)arrayget(arr, 0);
	off = duint32(sym, off, bv->n); // number of bits in each bitmap
	for(i = 0; i < len; i++) {
		// bitmap words
		bv = *(Bvec**)arrayget(arr, i);
		if(bv == nil)
			break;
		for(j = 0; j < bv->n; j += 32) {
			word = bv->b[j/32];
			// Runtime reads the bitmaps as byte arrays. Oblige.
			off = duint8(sym, off, word);
			off = duint8(sym, off, word>>8);
			off = duint8(sym, off, word>>16);
			off = duint8(sym, off, word>>24);
		}
	}
	duint32(sym, 0, i); // number of bitmaps
	ggloblsym(sym, off, RODATA);
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
liveness(Node *fn, Prog *firstp, Sym *argssym, Sym *livesym)
{
	Array *cfg, *vars;
	Liveness *lv;
	int debugdelta;
	NodeList *l;

	// Change name to dump debugging information only for a specific function.
	debugdelta = 0;
	if(strcmp(curfn->nname->sym->name, "!") == 0)
		debugdelta = 2;
	
	debuglive += debugdelta;
	if(debuglive >= 3) {
		print("liveness: %s\n", curfn->nname->sym->name);
		printprog(firstp);
	}
	checkptxt(fn, firstp);

	// Construct the global liveness state.
	cfg = newcfg(firstp);
	if(debuglive >= 3)
		printcfg(cfg);
	vars = getvariables(fn);
	lv = newliveness(fn, firstp, cfg, vars);

	// Run the dataflow framework.
	livenessprologue(lv);
	if(debuglive >= 3)
		livenessprintcfg(lv);
	livenesssolve(lv);
	if(debuglive >= 3)
		livenessprintcfg(lv);
	livenessepilogue(lv);
	if(debuglive >= 3)
		livenessprintcfg(lv);
	livenesscompact(lv);

	if(debuglive >= 2)
		livenessprintdebug(lv);

	// Emit the live pointer map data structures
	twobitwritesymbol(lv->livepointers, livesym);
	twobitwritesymbol(lv->argslivepointers, argssym);

	// Free everything.
	for(l=fn->dcl; l != nil; l = l->next)
		if(l->n != N)
			l->n->opt = nil;
	freeliveness(lv);
	arrayfree(vars);
	freecfg(cfg);
	
	debuglive -= debugdelta;
}
