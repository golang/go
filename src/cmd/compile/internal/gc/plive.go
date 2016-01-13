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

package gc

import (
	"cmd/internal/obj"
	"fmt"
	"sort"
)

const (
	UNVISITED = 0
	VISITED   = 1
)

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
type BasicBlock struct {
	pred            []*BasicBlock // predecessors; if none, probably start of CFG
	succ            []*BasicBlock // successors; if none, probably ends in return statement
	first           *obj.Prog     // first instruction in block
	last            *obj.Prog     // last instruction in block
	rpo             int           // reverse post-order number (also index in cfg)
	mark            int           // mark bit for traversals
	lastbitmapindex int           // for livenessepilogue

	// Summary sets of block effects.

	// Computed during livenessprologue using only the content of
	// individual blocks:
	//
	//	uevar: upward exposed variables (used before set in block)
	//	varkill: killed variables (set in block)
	//	avarinit: addrtaken variables set or used (proof of initialization)
	uevar    Bvec
	varkill  Bvec
	avarinit Bvec

	// Computed during livenesssolve using control flow information:
	//
	//	livein: variables live at block entry
	//	liveout: variables live at block exit
	//	avarinitany: addrtaken variables possibly initialized at block exit
	//		(initialized in block or at exit from any predecessor block)
	//	avarinitall: addrtaken variables certainly initialized at block exit
	//		(initialized in block or at exit from all predecessor blocks)
	livein      Bvec
	liveout     Bvec
	avarinitany Bvec
	avarinitall Bvec
}

// A collection of global state used by liveness analysis.
type Liveness struct {
	fn   *Node
	ptxt *obj.Prog
	vars []*Node
	cfg  []*BasicBlock

	// An array with a bit vector for each safe point tracking live pointers
	// in the arguments and locals area, indexed by bb.rpo.
	argslivepointers []Bvec
	livepointers     []Bvec
}

// Constructs a new basic block containing a single instruction.
func newblock(prog *obj.Prog) *BasicBlock {
	if prog == nil {
		Fatalf("newblock: prog cannot be nil")
	}
	result := new(BasicBlock)
	result.rpo = -1
	result.mark = UNVISITED
	result.first = prog
	result.last = prog
	result.pred = make([]*BasicBlock, 0, 2)
	result.succ = make([]*BasicBlock, 0, 2)
	return result
}

// Adds an edge between two basic blocks by making from a predecessor of to and
// to a successor of from.
func addedge(from *BasicBlock, to *BasicBlock) {
	if from == nil {
		Fatalf("addedge: from is nil")
	}
	if to == nil {
		Fatalf("addedge: to is nil")
	}
	from.succ = append(from.succ, to)
	to.pred = append(to.pred, from)
}

// Inserts prev before curr in the instruction
// stream.  Any control flow, such as branches or fall throughs, that target the
// existing instruction are adjusted to target the new instruction.
func splicebefore(lv *Liveness, bb *BasicBlock, prev *obj.Prog, curr *obj.Prog) {
	// There may be other instructions pointing at curr,
	// and we want them to now point at prev. Instead of
	// trying to find all such instructions, swap the contents
	// so that the problem becomes inserting next after curr.
	// The "opt" field is the backward link in the linked list.

	// Overwrite curr's data with prev, but keep the list links.
	tmp := *curr

	*curr = *prev
	curr.Opt = tmp.Opt
	curr.Link = tmp.Link

	// Overwrite prev (now next) with curr's old data.
	next := prev

	*next = tmp
	next.Opt = nil
	next.Link = nil

	// Now insert next after curr.
	next.Link = curr.Link

	next.Opt = curr
	curr.Link = next
	if next.Link != nil && next.Link.Opt == curr {
		next.Link.Opt = next
	}

	if bb.last == curr {
		bb.last = next
	}
}

// A pretty printer for basic blocks.
func printblock(bb *BasicBlock) {
	fmt.Printf("basic block %d\n", bb.rpo)
	fmt.Printf("\tpred:")
	for _, pred := range bb.pred {
		fmt.Printf(" %d", pred.rpo)
	}
	fmt.Printf("\n")
	fmt.Printf("\tsucc:")
	for _, succ := range bb.succ {
		fmt.Printf(" %d", succ.rpo)
	}
	fmt.Printf("\n")
	fmt.Printf("\tprog:\n")
	for prog := bb.first; ; prog = prog.Link {
		fmt.Printf("\t\t%v\n", prog)
		if prog == bb.last {
			break
		}
	}
}

// Iterates over a basic block applying a callback to each instruction.  There
// are two criteria for termination.  If the end of basic block is reached a
// value of zero is returned.  If the callback returns a non-zero value, the
// iteration is stopped and the value of the callback is returned.
func blockany(bb *BasicBlock, f func(*obj.Prog) bool) bool {
	for p := bb.last; p != nil; p = p.Opt.(*obj.Prog) {
		if f(p) {
			return true
		}
	}
	return false
}

// Collects and returns and array of Node*s for functions arguments and local
// variables.
func getvariables(fn *Node) []*Node {
	result := make([]*Node, 0, 0)
	for ll := fn.Func.Dcl; ll != nil; ll = ll.Next {
		if ll.N.Op == ONAME {
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
			ll.N.SetOpt(nil)

			// The compiler doesn't emit initializations for zero-width parameters or results.
			if ll.N.Type.Width == 0 {
				continue
			}

			ll.N.Name.Curfn = Curfn
			switch ll.N.Class {
			case PAUTO:
				if haspointers(ll.N.Type) {
					ll.N.SetOpt(int32(len(result)))
					result = append(result, ll.N)
				}

			case PPARAM, PPARAMOUT:
				ll.N.SetOpt(int32(len(result)))
				result = append(result, ll.N)
			}
		}
	}

	return result
}

// A pretty printer for control flow graphs.  Takes an array of BasicBlock*s.
func printcfg(cfg []*BasicBlock) {
	for _, bb := range cfg {
		printblock(bb)
	}
}

// Assigns a reverse post order number to each connected basic block using the
// standard algorithm.  Unconnected blocks will not be affected.
func reversepostorder(root *BasicBlock, rpo *int32) {
	root.mark = VISITED
	for _, bb := range root.succ {
		if bb.mark == UNVISITED {
			reversepostorder(bb, rpo)
		}
	}
	*rpo -= 1
	root.rpo = int(*rpo)
}

// Comparison predicate used for sorting basic blocks by their rpo in ascending
// order.
type blockrpocmp []*BasicBlock

func (x blockrpocmp) Len() int           { return len(x) }
func (x blockrpocmp) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x blockrpocmp) Less(i, j int) bool { return x[i].rpo < x[j].rpo }

// A pattern matcher for call instructions.  Returns true when the instruction
// is a call to a specific package qualified function name.
func iscall(prog *obj.Prog, name *obj.LSym) bool {
	if prog == nil {
		Fatalf("iscall: prog is nil")
	}
	if name == nil {
		Fatalf("iscall: function name is nil")
	}
	if prog.As != obj.ACALL {
		return false
	}
	return name == prog.To.Sym
}

// Returns true for instructions that call a runtime function implementing a
// select communication clause.

var selectNames [4]*obj.LSym

func isselectcommcasecall(prog *obj.Prog) bool {
	if selectNames[0] == nil {
		selectNames[0] = Linksym(Pkglookup("selectsend", Runtimepkg))
		selectNames[1] = Linksym(Pkglookup("selectrecv", Runtimepkg))
		selectNames[2] = Linksym(Pkglookup("selectrecv2", Runtimepkg))
		selectNames[3] = Linksym(Pkglookup("selectdefault", Runtimepkg))
	}

	for _, name := range selectNames {
		if iscall(prog, name) {
			return true
		}
	}
	return false
}

// Returns true for call instructions that target runtime·newselect.

var isnewselect_sym *obj.LSym

func isnewselect(prog *obj.Prog) bool {
	if isnewselect_sym == nil {
		isnewselect_sym = Linksym(Pkglookup("newselect", Runtimepkg))
	}
	return iscall(prog, isnewselect_sym)
}

// Returns true for call instructions that target runtime·selectgo.

var isselectgocall_sym *obj.LSym

func isselectgocall(prog *obj.Prog) bool {
	if isselectgocall_sym == nil {
		isselectgocall_sym = Linksym(Pkglookup("selectgo", Runtimepkg))
	}
	return iscall(prog, isselectgocall_sym)
}

var isdeferreturn_sym *obj.LSym

func isdeferreturn(prog *obj.Prog) bool {
	if isdeferreturn_sym == nil {
		isdeferreturn_sym = Linksym(Pkglookup("deferreturn", Runtimepkg))
	}
	return iscall(prog, isdeferreturn_sym)
}

// Walk backwards from a runtime·selectgo call up to its immediately dominating
// runtime·newselect call.  Any successor nodes of communication clause nodes
// are implicit successors of the runtime·selectgo call node.  The goal of this
// analysis is to add these missing edges to complete the control flow graph.
func addselectgosucc(selectgo *BasicBlock) {
	var succ *BasicBlock

	pred := selectgo
	for {
		if len(pred.pred) == 0 {
			Fatalf("selectgo does not have a newselect")
		}
		pred = pred.pred[0]
		if blockany(pred, isselectcommcasecall) {
			// A select comm case block should have exactly one
			// successor.
			if len(pred.succ) != 1 {
				Fatalf("select comm case has too many successors")
			}
			succ = pred.succ[0]

			// Its successor should have exactly two successors.
			// The drop through should flow to the selectgo block
			// and the branch should lead to the select case
			// statements block.
			if len(succ.succ) != 2 {
				Fatalf("select comm case successor has too many successors")
			}

			// Add the block as a successor of the selectgo block.
			addedge(selectgo, succ)
		}

		if blockany(pred, isnewselect) {
			// Reached the matching newselect.
			break
		}
	}
}

// The entry point for the missing selectgo control flow algorithm.  Takes an
// array of BasicBlock*s containing selectgo calls.
func fixselectgo(selectgo []*BasicBlock) {
	for _, bb := range selectgo {
		addselectgosucc(bb)
	}
}

// Constructs a control flow graph from a sequence of instructions.  This
// procedure is complicated by various sources of implicit control flow that are
// not accounted for using the standard cfg construction algorithm.  Returns an
// array of BasicBlock*s in control flow graph form (basic blocks ordered by
// their RPO number).
func newcfg(firstp *obj.Prog) []*BasicBlock {
	// Reset the opt field of each prog to nil.  In the first and second
	// passes, instructions that are labels temporarily use the opt field to
	// point to their basic block.  In the third pass, the opt field reset
	// to point to the predecessor of an instruction in its basic block.
	for p := firstp; p != nil; p = p.Link {
		p.Opt = nil
	}

	// Allocate an array to remember where we have seen selectgo calls.
	// These blocks will be revisited to add successor control flow edges.
	selectgo := make([]*BasicBlock, 0, 0)

	// Loop through all instructions identifying branch targets
	// and fall-throughs and allocate basic blocks.
	cfg := make([]*BasicBlock, 0, 0)

	bb := newblock(firstp)
	cfg = append(cfg, bb)
	for p := firstp; p != nil; p = p.Link {
		Thearch.Proginfo(p)
		if p.To.Type == obj.TYPE_BRANCH {
			if p.To.Val == nil {
				Fatalf("prog branch to nil")
			}
			if p.To.Val.(*obj.Prog).Opt == nil {
				p.To.Val.(*obj.Prog).Opt = newblock(p.To.Val.(*obj.Prog))
				cfg = append(cfg, p.To.Val.(*obj.Prog).Opt.(*BasicBlock))
			}

			if p.As != obj.AJMP && p.Link != nil && p.Link.Opt == nil {
				p.Link.Opt = newblock(p.Link)
				cfg = append(cfg, p.Link.Opt.(*BasicBlock))
			}
		} else if isselectcommcasecall(p) || isselectgocall(p) {
			// Accommodate implicit selectgo control flow.
			if p.Link.Opt == nil {
				p.Link.Opt = newblock(p.Link)
				cfg = append(cfg, p.Link.Opt.(*BasicBlock))
			}
		}
	}

	// Loop through all basic blocks maximally growing the list of
	// contained instructions until a label is reached.  Add edges
	// for branches and fall-through instructions.
	for _, bb := range cfg {
		for p := bb.last; p != nil; p = p.Link {
			if p.Opt != nil && p != bb.last {
				break
			}
			bb.last = p

			// Stop before an unreachable RET, to avoid creating
			// unreachable control flow nodes.
			if p.Link != nil && p.Link.As == obj.ARET && p.Link.Mode == 1 {
				break
			}

			// Collect basic blocks with selectgo calls.
			if isselectgocall(p) {
				selectgo = append(selectgo, bb)
			}
		}

		if bb.last.To.Type == obj.TYPE_BRANCH {
			addedge(bb, bb.last.To.Val.(*obj.Prog).Opt.(*BasicBlock))
		}
		if bb.last.Link != nil {
			// Add a fall-through when the instruction is
			// not an unconditional control transfer.
			if bb.last.As != obj.AJMP && bb.last.As != obj.ARET && bb.last.As != obj.AUNDEF {
				addedge(bb, bb.last.Link.Opt.(*BasicBlock))
			}
		}
	}

	// Add back links so the instructions in a basic block can be traversed
	// backward.  This is the final state of the instruction opt field.
	for _, bb := range cfg {
		p := bb.first
		var prev *obj.Prog
		for {
			p.Opt = prev
			if p == bb.last {
				break
			}
			prev = p
			p = p.Link
		}
	}

	// Add missing successor edges to the selectgo blocks.
	if len(selectgo) != 0 {
		fixselectgo([]*BasicBlock(selectgo))
	}

	// Find a depth-first order and assign a depth-first number to
	// all basic blocks.
	for _, bb := range cfg {
		bb.mark = UNVISITED
	}
	bb = cfg[0]
	rpo := int32(len(cfg))
	reversepostorder(bb, &rpo)

	// Sort the basic blocks by their depth first number.  The
	// array is now a depth-first spanning tree with the first
	// node being the root.
	sort.Sort(blockrpocmp(cfg))

	// Unreachable control flow nodes are indicated by a -1 in the rpo
	// field.  If we see these nodes something must have gone wrong in an
	// upstream compilation phase.
	bb = cfg[0]
	if bb.rpo == -1 {
		fmt.Printf("newcfg: unreachable basic block for %v\n", bb.last)
		printcfg(cfg)
		Fatalf("newcfg: invalid control flow graph")
	}

	return cfg
}

// Frees a control flow graph (an array of BasicBlock*s) and all of its leaf
// data structures.
func freecfg(cfg []*BasicBlock) {
	if len(cfg) > 0 {
		bb0 := cfg[0]
		for p := bb0.first; p != nil; p = p.Link {
			p.Opt = nil
		}
	}
}

// Returns true if the node names a variable that is otherwise uninteresting to
// the liveness computation.
func isfunny(n *Node) bool {
	return n.Sym != nil && (n.Sym.Name == ".fp" || n.Sym.Name == ".args")
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
func progeffects(prog *obj.Prog, vars []*Node, uevar Bvec, varkill Bvec, avarinit Bvec) {
	bvresetall(uevar)
	bvresetall(varkill)
	bvresetall(avarinit)

	if prog.As == obj.ARET {
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
		for i, node := range vars {
			switch node.Class &^ PHEAP {
			case PPARAM:
				bvset(uevar, int32(i))

				// If the result had its address taken, it is being tracked
			// by the avarinit code, which does not use uevar.
			// If we added it to uevar too, we'd not see any kill
			// and decide that the variable was live entry, which it is not.
			// So only use uevar in the non-addrtaken case.
			// The p->to.type == thearch.D_NONE limits the bvset to
			// non-tail-call return instructions; see note above
			// the for loop for details.
			case PPARAMOUT:
				if !node.Addrtaken && prog.To.Type == obj.TYPE_NONE {
					bvset(uevar, int32(i))
				}
			}
		}

		return
	}

	if prog.As == obj.ATEXT {
		// A text instruction marks the entry point to a function and
		// the definition point of all in arguments.
		for i, node := range vars {
			switch node.Class &^ PHEAP {
			case PPARAM:
				if node.Addrtaken {
					bvset(avarinit, int32(i))
				}
				bvset(varkill, int32(i))
			}
		}

		return
	}

	if prog.Info.Flags&(LeftRead|LeftWrite|LeftAddr) != 0 {
		from := &prog.From
		if from.Node != nil && from.Sym != nil && ((from.Node).(*Node)).Name.Curfn == Curfn {
			switch ((from.Node).(*Node)).Class &^ PHEAP {
			case PAUTO, PPARAM, PPARAMOUT:
				pos, ok := from.Node.(*Node).Opt().(int32) // index in vars
				if !ok {
					goto Next
				}
				if pos >= int32(len(vars)) || vars[pos] != from.Node {
					Fatalf("bad bookkeeping in liveness %v %d", Nconv(from.Node.(*Node), 0), pos)
				}
				if ((from.Node).(*Node)).Addrtaken {
					bvset(avarinit, pos)
				} else {
					if prog.Info.Flags&(LeftRead|LeftAddr) != 0 {
						bvset(uevar, pos)
					}
					if prog.Info.Flags&LeftWrite != 0 {
						if from.Node != nil && !Isfat(((from.Node).(*Node)).Type) {
							bvset(varkill, pos)
						}
					}
				}
			}
		}
	}

Next:
	if prog.Info.Flags&(RightRead|RightWrite|RightAddr) != 0 {
		to := &prog.To
		if to.Node != nil && to.Sym != nil && ((to.Node).(*Node)).Name.Curfn == Curfn {
			switch ((to.Node).(*Node)).Class &^ PHEAP {
			case PAUTO, PPARAM, PPARAMOUT:
				pos, ok := to.Node.(*Node).Opt().(int32) // index in vars
				if !ok {
					return
				}
				if pos >= int32(len(vars)) || vars[pos] != to.Node {
					Fatalf("bad bookkeeping in liveness %v %d", Nconv(to.Node.(*Node), 0), pos)
				}
				if ((to.Node).(*Node)).Addrtaken {
					if prog.As != obj.AVARKILL {
						bvset(avarinit, pos)
					}
					if prog.As == obj.AVARDEF || prog.As == obj.AVARKILL {
						bvset(varkill, pos)
					}
				} else {
					// RightRead is a read, obviously.
					// RightAddr by itself is also implicitly a read.
					//
					// RightAddr|RightWrite means that the address is being taken
					// but only so that the instruction can write to the value.
					// It is not a read. It is equivalent to RightWrite except that
					// having the RightAddr bit set keeps the registerizer from
					// trying to substitute a register for the memory location.
					if (prog.Info.Flags&RightRead != 0) || prog.Info.Flags&(RightAddr|RightWrite) == RightAddr {
						bvset(uevar, pos)
					}
					if prog.Info.Flags&RightWrite != 0 {
						if to.Node != nil && (!Isfat(((to.Node).(*Node)).Type) || prog.As == obj.AVARDEF) {
							bvset(varkill, pos)
						}
					}
				}
			}
		}
	}
}

// Constructs a new liveness structure used to hold the global state of the
// liveness computation.  The cfg argument is an array of BasicBlock*s and the
// vars argument is an array of Node*s.
func newliveness(fn *Node, ptxt *obj.Prog, cfg []*BasicBlock, vars []*Node) *Liveness {
	result := new(Liveness)
	result.fn = fn
	result.ptxt = ptxt
	result.cfg = cfg
	result.vars = vars

	nblocks := int32(len(cfg))
	nvars := int32(len(vars))
	bulk := bvbulkalloc(nvars, nblocks*7)
	for _, bb := range cfg {
		bb.uevar = bulk.next()
		bb.varkill = bulk.next()
		bb.livein = bulk.next()
		bb.liveout = bulk.next()
		bb.avarinit = bulk.next()
		bb.avarinitany = bulk.next()
		bb.avarinitall = bulk.next()
	}

	result.livepointers = make([]Bvec, 0, 0)
	result.argslivepointers = make([]Bvec, 0, 0)
	return result
}

// Frees the liveness structure and all of its leaf data structures.
func freeliveness(lv *Liveness) {
	if lv == nil {
		Fatalf("freeliveness: cannot free nil")
	}
}

func printeffects(p *obj.Prog, uevar Bvec, varkill Bvec, avarinit Bvec) {
	fmt.Printf("effects of %v", p)
	fmt.Printf("\nuevar: ")
	bvprint(uevar)
	fmt.Printf("\nvarkill: ")
	bvprint(varkill)
	fmt.Printf("\navarinit: ")
	bvprint(avarinit)
	fmt.Printf("\n")
}

// Pretty print a variable node.  Uses Pascal like conventions for pointers and
// addresses to avoid confusing the C like conventions used in the node variable
// names.
func printnode(node *Node) {
	p := ""
	if haspointers(node.Type) {
		p = "^"
	}
	a := ""
	if node.Addrtaken {
		a = "@"
	}
	fmt.Printf(" %v%s%s", node, p, a)
}

// Pretty print a list of variables.  The vars argument is an array of Node*s.
func printvars(name string, bv Bvec, vars []*Node) {
	fmt.Printf("%s:", name)
	for i, node := range vars {
		if bvget(bv, int32(i)) != 0 {
			printnode(node)
		}
	}
	fmt.Printf("\n")
}

// Prints a basic block annotated with the information computed by liveness
// analysis.
func livenessprintblock(lv *Liveness, bb *BasicBlock) {
	fmt.Printf("basic block %d\n", bb.rpo)

	fmt.Printf("\tpred:")
	for _, pred := range bb.pred {
		fmt.Printf(" %d", pred.rpo)
	}
	fmt.Printf("\n")

	fmt.Printf("\tsucc:")
	for _, succ := range bb.succ {
		fmt.Printf(" %d", succ.rpo)
	}
	fmt.Printf("\n")

	printvars("\tuevar", bb.uevar, []*Node(lv.vars))
	printvars("\tvarkill", bb.varkill, []*Node(lv.vars))
	printvars("\tlivein", bb.livein, []*Node(lv.vars))
	printvars("\tliveout", bb.liveout, []*Node(lv.vars))
	printvars("\tavarinit", bb.avarinit, []*Node(lv.vars))
	printvars("\tavarinitany", bb.avarinitany, []*Node(lv.vars))
	printvars("\tavarinitall", bb.avarinitall, []*Node(lv.vars))

	fmt.Printf("\tprog:\n")
	for prog := bb.first; ; prog = prog.Link {
		fmt.Printf("\t\t%v", prog)
		if prog.As == obj.APCDATA && prog.From.Offset == obj.PCDATA_StackMapIndex {
			pos := int32(prog.To.Offset)
			live := lv.livepointers[pos]
			fmt.Printf(" ")
			bvprint(live)
		}

		fmt.Printf("\n")
		if prog == bb.last {
			break
		}
	}
}

// Prints a control flow graph annotated with any information computed by
// liveness analysis.
func livenessprintcfg(lv *Liveness) {
	for _, bb := range lv.cfg {
		livenessprintblock(lv, bb)
	}
}

func checkauto(fn *Node, p *obj.Prog, n *Node) {
	for l := fn.Func.Dcl; l != nil; l = l.Next {
		if l.N.Op == ONAME && l.N.Class == PAUTO && l.N == n {
			return
		}
	}

	if n == nil {
		fmt.Printf("%v: checkauto %v: nil node in %v\n", p.Line(), Curfn, p)
		return
	}

	fmt.Printf("checkauto %v: %v (%p; class=%d) not found in %p %v\n", funcSym(Curfn), n, n, n.Class, p, p)
	for l := fn.Func.Dcl; l != nil; l = l.Next {
		fmt.Printf("\t%v (%p; class=%d)\n", l.N, l.N, l.N.Class)
	}
	Yyerror("checkauto: invariant lost")
}

func checkparam(fn *Node, p *obj.Prog, n *Node) {
	if isfunny(n) {
		return
	}
	var a *Node
	var class Class
	for l := fn.Func.Dcl; l != nil; l = l.Next {
		a = l.N
		class = a.Class &^ PHEAP
		if a.Op == ONAME && (class == PPARAM || class == PPARAMOUT) && a == n {
			return
		}
	}

	fmt.Printf("checkparam %v: %v (%p; class=%d) not found in %v\n", Curfn, n, n, n.Class, p)
	for l := fn.Func.Dcl; l != nil; l = l.Next {
		fmt.Printf("\t%v (%p; class=%d)\n", l.N, l.N, l.N.Class)
	}
	Yyerror("checkparam: invariant lost")
}

func checkprog(fn *Node, p *obj.Prog) {
	if p.From.Name == obj.NAME_AUTO {
		checkauto(fn, p, p.From.Node.(*Node))
	}
	if p.From.Name == obj.NAME_PARAM {
		checkparam(fn, p, p.From.Node.(*Node))
	}
	if p.To.Name == obj.NAME_AUTO {
		checkauto(fn, p, p.To.Node.(*Node))
	}
	if p.To.Name == obj.NAME_PARAM {
		checkparam(fn, p, p.To.Node.(*Node))
	}
}

// Check instruction invariants.  We assume that the nodes corresponding to the
// sources and destinations of memory operations will be declared in the
// function.  This is not strictly true, as is the case for the so-called funny
// nodes and there are special cases to skip over that stuff.  The analysis will
// fail if this invariant blindly changes.
func checkptxt(fn *Node, firstp *obj.Prog) {
	if debuglive == 0 {
		return
	}

	for p := firstp; p != nil; p = p.Link {
		if false {
			fmt.Printf("analyzing '%v'\n", p)
		}
		if p.As != obj.ADATA && p.As != obj.AGLOBL && p.As != obj.ATYPE {
			checkprog(fn, p)
		}
	}
}

// NOTE: The bitmap for a specific type t should be cached in t after the first run
// and then simply copied into bv at the correct offset on future calls with
// the same type t. On https://rsc.googlecode.com/hg/testdata/slow.go, onebitwalktype1
// accounts for 40% of the 6g execution time.
func onebitwalktype1(t *Type, xoffset *int64, bv Bvec) {
	if t.Align > 0 && *xoffset&int64(t.Align-1) != 0 {
		Fatalf("onebitwalktype1: invalid initial alignment, %v", t)
	}

	switch t.Etype {
	case TINT8,
		TUINT8,
		TINT16,
		TUINT16,
		TINT32,
		TUINT32,
		TINT64,
		TUINT64,
		TINT,
		TUINT,
		TUINTPTR,
		TBOOL,
		TFLOAT32,
		TFLOAT64,
		TCOMPLEX64,
		TCOMPLEX128:
		*xoffset += t.Width

	case TPTR32,
		TPTR64,
		TUNSAFEPTR,
		TFUNC,
		TCHAN,
		TMAP:
		if *xoffset&int64(Widthptr-1) != 0 {
			Fatalf("onebitwalktype1: invalid alignment, %v", t)
		}
		bvset(bv, int32(*xoffset/int64(Widthptr))) // pointer
		*xoffset += t.Width

	case TSTRING:
		// struct { byte *str; intgo len; }
		if *xoffset&int64(Widthptr-1) != 0 {
			Fatalf("onebitwalktype1: invalid alignment, %v", t)
		}
		bvset(bv, int32(*xoffset/int64(Widthptr))) //pointer in first slot
		*xoffset += t.Width

	case TINTER:
		// struct { Itab *tab;	void *data; }
		// or, when isnilinter(t)==true:
		// struct { Type *type; void *data; }
		if *xoffset&int64(Widthptr-1) != 0 {
			Fatalf("onebitwalktype1: invalid alignment, %v", t)
		}
		bvset(bv, int32(*xoffset/int64(Widthptr)))   // pointer in first slot
		bvset(bv, int32(*xoffset/int64(Widthptr)+1)) // pointer in second slot
		*xoffset += t.Width

	case TARRAY:
		// The value of t->bound is -1 for slices types and >=0 for
		// for fixed array types.  All other values are invalid.
		if t.Bound < -1 {
			Fatalf("onebitwalktype1: invalid bound, %v", t)
		}
		if Isslice(t) {
			// struct { byte *array; uintgo len; uintgo cap; }
			if *xoffset&int64(Widthptr-1) != 0 {
				Fatalf("onebitwalktype1: invalid TARRAY alignment, %v", t)
			}
			bvset(bv, int32(*xoffset/int64(Widthptr))) // pointer in first slot (BitsPointer)
			*xoffset += t.Width
		} else {
			for i := int64(0); i < t.Bound; i++ {
				onebitwalktype1(t.Type, xoffset, bv)
			}
		}

	case TSTRUCT:
		o := int64(0)
		var fieldoffset int64
		for t1 := t.Type; t1 != nil; t1 = t1.Down {
			fieldoffset = t1.Width
			*xoffset += fieldoffset - o
			onebitwalktype1(t1.Type, xoffset, bv)
			o = fieldoffset + t1.Type.Width
		}

		*xoffset += t.Width - o

	default:
		Fatalf("onebitwalktype1: unexpected type, %v", t)
	}
}

// Returns the number of words of local variables.
func localswords() int32 {
	return int32(stkptrsize / int64(Widthptr))
}

// Returns the number of words of in and out arguments.
func argswords() int32 {
	return int32(Curfn.Type.Argwid / int64(Widthptr))
}

// Generates live pointer value maps for arguments and local variables.  The
// this argument and the in arguments are always assumed live.  The vars
// argument is an array of Node*s.
func onebitlivepointermap(lv *Liveness, liveout Bvec, vars []*Node, args Bvec, locals Bvec) {
	var node *Node
	var xoffset int64

	for i := int32(0); ; i++ {
		i = int32(bvnext(liveout, i))
		if i < 0 {
			break
		}
		node = vars[i]
		switch node.Class {
		case PAUTO:
			xoffset = node.Xoffset + stkptrsize
			onebitwalktype1(node.Type, &xoffset, locals)

		case PPARAM, PPARAMOUT:
			xoffset = node.Xoffset
			onebitwalktype1(node.Type, &xoffset, args)
		}
	}

	// The node list only contains declared names.
	// If the receiver or arguments are unnamed, they will be omitted
	// from the list above. Preserve those values - even though they are unused -
	// in order to keep their addresses live for use in stack traces.
	thisargtype := getthisx(lv.fn.Type)

	if thisargtype != nil {
		xoffset = 0
		onebitwalktype1(thisargtype, &xoffset, args)
	}

	inargtype := getinargx(lv.fn.Type)
	if inargtype != nil {
		xoffset = 0
		onebitwalktype1(inargtype, &xoffset, args)
	}
}

// Construct a disembodied instruction.
func unlinkedprog(as int) *obj.Prog {
	p := Ctxt.NewProg()
	Clearp(p)
	p.As = int16(as)
	return p
}

// Construct a new PCDATA instruction associated with and for the purposes of
// covering an existing instruction.
func newpcdataprog(prog *obj.Prog, index int32) *obj.Prog {
	var from Node
	var to Node

	Nodconst(&from, Types[TINT32], obj.PCDATA_StackMapIndex)
	Nodconst(&to, Types[TINT32], int64(index))
	pcdata := unlinkedprog(obj.APCDATA)
	pcdata.Lineno = prog.Lineno
	Naddr(&pcdata.From, &from)
	Naddr(&pcdata.To, &to)
	return pcdata
}

// Returns true for instructions that are safe points that must be annotated
// with liveness information.
func issafepoint(prog *obj.Prog) bool {
	return prog.As == obj.ATEXT || prog.As == obj.ACALL
}

// Initializes the sets for solving the live variables.  Visits all the
// instructions in each basic block to summarizes the information at each basic
// block
func livenessprologue(lv *Liveness) {
	nvars := int32(len(lv.vars))
	uevar := bvalloc(nvars)
	varkill := bvalloc(nvars)
	avarinit := bvalloc(nvars)
	for _, bb := range lv.cfg {
		// Walk the block instructions backward and update the block
		// effects with the each prog effects.
		for p := bb.last; p != nil; p = p.Opt.(*obj.Prog) {
			progeffects(p, []*Node(lv.vars), uevar, varkill, avarinit)
			if debuglive >= 3 {
				printeffects(p, uevar, varkill, avarinit)
			}
			bvor(bb.varkill, bb.varkill, varkill)
			bvandnot(bb.uevar, bb.uevar, varkill)
			bvor(bb.uevar, bb.uevar, uevar)
		}

		// Walk the block instructions forward to update avarinit bits.
		// avarinit describes the effect at the end of the block, not the beginning.
		bvresetall(varkill)

		for p := bb.first; ; p = p.Link {
			progeffects(p, []*Node(lv.vars), uevar, varkill, avarinit)
			if debuglive >= 3 {
				printeffects(p, uevar, varkill, avarinit)
			}
			bvandnot(bb.avarinit, bb.avarinit, varkill)
			bvor(bb.avarinit, bb.avarinit, avarinit)
			if p == bb.last {
				break
			}
		}
	}
}

// Solve the liveness dataflow equations.
func livenesssolve(lv *Liveness) {
	// These temporary bitvectors exist to avoid successive allocations and
	// frees within the loop.
	newlivein := bvalloc(int32(len(lv.vars)))

	newliveout := bvalloc(int32(len(lv.vars)))
	any := bvalloc(int32(len(lv.vars)))
	all := bvalloc(int32(len(lv.vars)))

	// Push avarinitall, avarinitany forward.
	// avarinitall says the addressed var is initialized along all paths reaching the block exit.
	// avarinitany says the addressed var is initialized along some path reaching the block exit.
	for i, bb := range lv.cfg {
		if i == 0 {
			bvcopy(bb.avarinitall, bb.avarinit)
		} else {
			bvresetall(bb.avarinitall)
			bvnot(bb.avarinitall)
		}
		bvcopy(bb.avarinitany, bb.avarinit)
	}

	change := int32(1)
	for change != 0 {
		change = 0
		for _, bb := range lv.cfg {
			bvresetall(any)
			bvresetall(all)
			for j, pred := range bb.pred {
				if j == 0 {
					bvcopy(any, pred.avarinitany)
					bvcopy(all, pred.avarinitall)
				} else {
					bvor(any, any, pred.avarinitany)
					bvand(all, all, pred.avarinitall)
				}
			}

			bvandnot(any, any, bb.varkill)
			bvandnot(all, all, bb.varkill)
			bvor(any, any, bb.avarinit)
			bvor(all, all, bb.avarinit)
			if bvcmp(any, bb.avarinitany) != 0 {
				change = 1
				bvcopy(bb.avarinitany, any)
			}

			if bvcmp(all, bb.avarinitall) != 0 {
				change = 1
				bvcopy(bb.avarinitall, all)
			}
		}
	}

	// Iterate through the blocks in reverse round-robin fashion.  A work
	// queue might be slightly faster.  As is, the number of iterations is
	// so low that it hardly seems to be worth the complexity.
	change = 1

	for change != 0 {
		change = 0

		// Walk blocks in the general direction of propagation.  This
		// improves convergence.
		for i := len(lv.cfg) - 1; i >= 0; i-- {
			bb := lv.cfg[i]

			// A variable is live on output from this block
			// if it is live on input to some successor.
			//
			// out[b] = \bigcup_{s \in succ[b]} in[s]
			bvresetall(newliveout)
			for _, succ := range bb.succ {
				bvor(newliveout, newliveout, succ.livein)
			}

			if bvcmp(bb.liveout, newliveout) != 0 {
				change = 1
				bvcopy(bb.liveout, newliveout)
			}

			// A variable is live on input to this block
			// if it is live on output from this block and
			// not set by the code in this block.
			//
			// in[b] = uevar[b] \cup (out[b] \setminus varkill[b])
			bvandnot(newlivein, bb.liveout, bb.varkill)

			bvor(bb.livein, newlivein, bb.uevar)
		}
	}
}

// This function is slow but it is only used for generating debug prints.
// Check whether n is marked live in args/locals.
func islive(n *Node, args Bvec, locals Bvec) bool {
	switch n.Class {
	case PPARAM, PPARAMOUT:
		for i := 0; int64(i) < n.Type.Width/int64(Widthptr); i++ {
			if bvget(args, int32(n.Xoffset/int64(Widthptr)+int64(i))) != 0 {
				return true
			}
		}

	case PAUTO:
		for i := 0; int64(i) < n.Type.Width/int64(Widthptr); i++ {
			if bvget(locals, int32((n.Xoffset+stkptrsize)/int64(Widthptr)+int64(i))) != 0 {
				return true
			}
		}
	}

	return false
}

// Visits all instructions in a basic block and computes a bit vector of live
// variables at each safe point locations.
func livenessepilogue(lv *Liveness) {
	var pred *BasicBlock
	var args Bvec
	var locals Bvec
	var n *Node
	var p *obj.Prog
	var j int32
	var pos int32
	var xoffset int64

	nvars := int32(len(lv.vars))
	livein := bvalloc(nvars)
	liveout := bvalloc(nvars)
	uevar := bvalloc(nvars)
	varkill := bvalloc(nvars)
	avarinit := bvalloc(nvars)
	any := bvalloc(nvars)
	all := bvalloc(nvars)
	ambig := bvalloc(localswords())
	nmsg := int32(0)
	startmsg := int32(0)

	for _, bb := range lv.cfg {
		// Compute avarinitany and avarinitall for entry to block.
		// This duplicates information known during livenesssolve
		// but avoids storing two more vectors for each block.
		bvresetall(any)

		bvresetall(all)
		for j = 0; j < int32(len(bb.pred)); j++ {
			pred = bb.pred[j]
			if j == 0 {
				bvcopy(any, pred.avarinitany)
				bvcopy(all, pred.avarinitall)
			} else {
				bvor(any, any, pred.avarinitany)
				bvand(all, all, pred.avarinitall)
			}
		}

		// Walk forward through the basic block instructions and
		// allocate liveness maps for those instructions that need them.
		// Seed the maps with information about the addrtaken variables.
		for p = bb.first; ; p = p.Link {
			progeffects(p, []*Node(lv.vars), uevar, varkill, avarinit)
			bvandnot(any, any, varkill)
			bvandnot(all, all, varkill)
			bvor(any, any, avarinit)
			bvor(all, all, avarinit)

			if issafepoint(p) {
				// Annotate ambiguously live variables so that they can
				// be zeroed at function entry.
				// livein and liveout are dead here and used as temporaries.
				bvresetall(livein)

				bvandnot(liveout, any, all)
				if !bvisempty(liveout) {
					for pos = 0; pos < liveout.n; pos++ {
						if bvget(liveout, pos) == 0 {
							continue
						}
						bvset(all, pos) // silence future warnings in this block
						n = lv.vars[pos]
						if !n.Name.Needzero {
							n.Name.Needzero = true
							if debuglive >= 1 {
								Warnl(int(p.Lineno), "%v: %v is ambiguously live", Curfn.Func.Nname, Nconv(n, obj.FmtLong))
							}

							// Record in 'ambiguous' bitmap.
							xoffset = n.Xoffset + stkptrsize

							onebitwalktype1(n.Type, &xoffset, ambig)
						}
					}
				}

				// Allocate a bit vector for each class and facet of
				// value we are tracking.

				// Live stuff first.
				args = bvalloc(argswords())

				lv.argslivepointers = append(lv.argslivepointers, args)
				locals = bvalloc(localswords())
				lv.livepointers = append(lv.livepointers, locals)

				if debuglive >= 3 {
					fmt.Printf("%v\n", p)
					printvars("avarinitany", any, lv.vars)
				}

				// Record any values with an "address taken" reaching
				// this code position as live. Must do now instead of below
				// because the any/all calculation requires walking forward
				// over the block (as this loop does), while the liveout
				// requires walking backward (as the next loop does).
				onebitlivepointermap(lv, any, lv.vars, args, locals)
			}

			if p == bb.last {
				break
			}
		}

		bb.lastbitmapindex = len(lv.livepointers) - 1
	}

	var fmt_ string
	var next *obj.Prog
	var numlive int32
	var msg []string
	for _, bb := range lv.cfg {
		if debuglive >= 1 && Curfn.Func.Nname.Sym.Name != "init" && Curfn.Func.Nname.Sym.Name[0] != '.' {
			nmsg = int32(len(lv.livepointers))
			startmsg = nmsg
			msg = make([]string, nmsg)
			for j = 0; j < nmsg; j++ {
				msg[j] = ""
			}
		}

		// walk backward, emit pcdata and populate the maps
		pos = int32(bb.lastbitmapindex)

		if pos < 0 {
			// the first block we encounter should have the ATEXT so
			// at no point should pos ever be less than zero.
			Fatalf("livenessepilogue")
		}

		bvcopy(livein, bb.liveout)
		for p = bb.last; p != nil; p = next {
			next = p.Opt.(*obj.Prog) // splicebefore modifies p->opt

			// Propagate liveness information
			progeffects(p, lv.vars, uevar, varkill, avarinit)

			bvcopy(liveout, livein)
			bvandnot(livein, liveout, varkill)
			bvor(livein, livein, uevar)
			if debuglive >= 3 && issafepoint(p) {
				fmt.Printf("%v\n", p)
				printvars("uevar", uevar, lv.vars)
				printvars("varkill", varkill, lv.vars)
				printvars("livein", livein, lv.vars)
				printvars("liveout", liveout, lv.vars)
			}

			if issafepoint(p) {
				// Found an interesting instruction, record the
				// corresponding liveness information.

				// Useful sanity check: on entry to the function,
				// the only things that can possibly be live are the
				// input parameters.
				if p.As == obj.ATEXT {
					for j = 0; j < liveout.n; j++ {
						if bvget(liveout, j) == 0 {
							continue
						}
						n = lv.vars[j]
						if n.Class != PPARAM {
							yyerrorl(int(p.Lineno), "internal error: %v %v recorded as live on entry", Curfn.Func.Nname, Nconv(n, obj.FmtLong))
						}
					}
				}

				// Record live pointers.
				args = lv.argslivepointers[pos]

				locals = lv.livepointers[pos]
				onebitlivepointermap(lv, liveout, lv.vars, args, locals)

				// Ambiguously live variables are zeroed immediately after
				// function entry. Mark them live for all the non-entry bitmaps
				// so that GODEBUG=gcdead=1 mode does not poison them.
				if p.As == obj.ACALL {
					bvor(locals, locals, ambig)
				}

				// Show live pointer bitmaps.
				// We're interpreting the args and locals bitmap instead of liveout so that we
				// include the bits added by the avarinit logic in the
				// previous loop.
				if msg != nil {
					fmt_ = ""
					fmt_ += fmt.Sprintf("%v: live at ", p.Line())
					if p.As == obj.ACALL && p.To.Node != nil {
						fmt_ += fmt.Sprintf("call to %s:", ((p.To.Node).(*Node)).Sym.Name)
					} else if p.As == obj.ACALL {
						fmt_ += "indirect call:"
					} else {
						fmt_ += fmt.Sprintf("entry to %s:", ((p.From.Node).(*Node)).Sym.Name)
					}
					numlive = 0
					for j = 0; j < int32(len(lv.vars)); j++ {
						n = lv.vars[j]
						if islive(n, args, locals) {
							fmt_ += fmt.Sprintf(" %v", n)
							numlive++
						}
					}

					fmt_ += "\n"
					if numlive == 0 { // squelch message

					} else {
						startmsg--
						msg[startmsg] = fmt_
					}
				}

				// Only CALL instructions need a PCDATA annotation.
				// The TEXT instruction annotation is implicit.
				if p.As == obj.ACALL {
					if isdeferreturn(p) {
						// runtime.deferreturn modifies its return address to return
						// back to the CALL, not to the subsequent instruction.
						// Because the return comes back one instruction early,
						// the PCDATA must begin one instruction early too.
						// The instruction before a call to deferreturn is always a
						// no-op, to keep PC-specific data unambiguous.
						prev := p.Opt.(*obj.Prog)
						if Ctxt.Arch.Thechar == '9' {
							// On ppc64 there is an additional instruction
							// (another no-op or reload of toc pointer) before
							// the call.
							prev = prev.Opt.(*obj.Prog)
						}
						splicebefore(lv, bb, newpcdataprog(prev, pos), prev)
					} else {
						splicebefore(lv, bb, newpcdataprog(p, pos), p)
					}
				}

				pos--
			}
		}

		if msg != nil {
			for j = startmsg; j < nmsg; j++ {
				if msg[j] != "" {
					fmt.Printf("%s", msg[j])
				}
			}

			msg = nil
			nmsg = 0
			startmsg = 0
		}
	}

	Flusherrors()
}

// FNV-1 hash function constants.
const (
	H0 = 2166136261
	Hp = 16777619
)

func hashbitmap(h uint32, bv Bvec) uint32 {
	var w uint32

	n := int((bv.n + 31) / 32)
	for i := 0; i < n; i++ {
		w = bv.b[i]
		h = (h * Hp) ^ (w & 0xff)
		h = (h * Hp) ^ ((w >> 8) & 0xff)
		h = (h * Hp) ^ ((w >> 16) & 0xff)
		h = (h * Hp) ^ ((w >> 24) & 0xff)
	}

	return h
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
func livenesscompact(lv *Liveness) {
	// Linear probing hash table of bitmaps seen so far.
	// The hash table has 4n entries to keep the linear
	// scan short. An entry of -1 indicates an empty slot.
	n := len(lv.livepointers)

	tablesize := 4 * n
	table := make([]int, tablesize)
	for i := range table {
		table[i] = -1
	}

	// remap[i] = the new index of the old bit vector #i.
	remap := make([]int, n)

	for i := range remap {
		remap[i] = -1
	}
	uniq := 0 // unique tables found so far

	// Consider bit vectors in turn.
	// If new, assign next number using uniq,
	// record in remap, record in lv->livepointers and lv->argslivepointers
	// under the new index, and add entry to hash table.
	// If already seen, record earlier index in remap and free bitmaps.
	var jarg Bvec
	var j int
	var h uint32
	var arg Bvec
	var jlocal Bvec
	var local Bvec
	for i := 0; i < n; i++ {
		local = lv.livepointers[i]
		arg = lv.argslivepointers[i]
		h = hashbitmap(hashbitmap(H0, local), arg) % uint32(tablesize)

		for {
			j = table[h]
			if j < 0 {
				break
			}
			jlocal = lv.livepointers[j]
			jarg = lv.argslivepointers[j]
			if bvcmp(local, jlocal) == 0 && bvcmp(arg, jarg) == 0 {
				remap[i] = j
				goto Next
			}

			h++
			if h == uint32(tablesize) {
				h = 0
			}
		}

		table[h] = uniq
		remap[i] = uniq
		lv.livepointers[uniq] = local
		lv.argslivepointers[uniq] = arg
		uniq++
	Next:
	}

	// We've already reordered lv->livepointers[0:uniq]
	// and lv->argslivepointers[0:uniq] and freed the bitmaps
	// we don't need anymore. Clear the pointers later in the
	// array so that we can tell where the coalesced bitmaps stop
	// and so that we don't double-free when cleaning up.
	for j := uniq; j < n; j++ {
		lv.livepointers[j] = Bvec{}
		lv.argslivepointers[j] = Bvec{}
	}

	// Rewrite PCDATA instructions to use new numbering.
	var i int
	for p := lv.ptxt; p != nil; p = p.Link {
		if p.As == obj.APCDATA && p.From.Offset == obj.PCDATA_StackMapIndex {
			i = int(p.To.Offset)
			if i >= 0 {
				p.To.Offset = int64(remap[i])
			}
		}
	}
}

func printbitset(printed int, name string, vars []*Node, bits Bvec) int {
	started := 0
	for i, n := range vars {
		if bvget(bits, int32(i)) == 0 {
			continue
		}
		if started == 0 {
			if printed == 0 {
				fmt.Printf("\t")
			} else {
				fmt.Printf(" ")
			}
			started = 1
			printed = 1
			fmt.Printf("%s=", name)
		} else {
			fmt.Printf(",")
		}

		fmt.Printf("%s", n.Sym.Name)
	}

	return printed
}

// Prints the computed liveness information and inputs, for debugging.
// This format synthesizes the information used during the multiple passes
// into a single presentation.
func livenessprintdebug(lv *Liveness) {
	var j int
	var printed int
	var p *obj.Prog
	var args Bvec
	var locals Bvec
	var n *Node

	fmt.Printf("liveness: %s\n", Curfn.Func.Nname.Sym.Name)

	uevar := bvalloc(int32(len(lv.vars)))
	varkill := bvalloc(int32(len(lv.vars)))
	avarinit := bvalloc(int32(len(lv.vars)))

	pcdata := 0
	for i, bb := range lv.cfg {
		if i > 0 {
			fmt.Printf("\n")
		}

		// bb#0 pred=1,2 succ=3,4
		fmt.Printf("bb#%d pred=", i)

		for j = 0; j < len(bb.pred); j++ {
			if j > 0 {
				fmt.Printf(",")
			}
			fmt.Printf("%d", (bb.pred[j]).rpo)
		}

		fmt.Printf(" succ=")
		for j = 0; j < len(bb.succ); j++ {
			if j > 0 {
				fmt.Printf(",")
			}
			fmt.Printf("%d", (bb.succ[j]).rpo)
		}

		fmt.Printf("\n")

		// initial settings
		printed = 0

		printed = printbitset(printed, "uevar", lv.vars, bb.uevar)
		printed = printbitset(printed, "livein", lv.vars, bb.livein)
		if printed != 0 {
			fmt.Printf("\n")
		}

		// program listing, with individual effects listed
		for p = bb.first; ; p = p.Link {
			fmt.Printf("%v\n", p)
			if p.As == obj.APCDATA && p.From.Offset == obj.PCDATA_StackMapIndex {
				pcdata = int(p.To.Offset)
			}
			progeffects(p, lv.vars, uevar, varkill, avarinit)
			printed = 0
			printed = printbitset(printed, "uevar", lv.vars, uevar)
			printed = printbitset(printed, "varkill", lv.vars, varkill)
			printed = printbitset(printed, "avarinit", lv.vars, avarinit)
			if printed != 0 {
				fmt.Printf("\n")
			}
			if issafepoint(p) {
				args = lv.argslivepointers[pcdata]
				locals = lv.livepointers[pcdata]
				fmt.Printf("\tlive=")
				printed = 0
				for j = 0; j < len(lv.vars); j++ {
					n = lv.vars[j]
					if islive(n, args, locals) {
						if printed != 0 {
							fmt.Printf(",")
						}
						fmt.Printf("%v", n)
						printed++
					}
				}
				fmt.Printf("\n")
			}

			if p == bb.last {
				break
			}
		}

		// bb bitsets
		fmt.Printf("end\n")

		printed = printbitset(printed, "varkill", lv.vars, bb.varkill)
		printed = printbitset(printed, "liveout", lv.vars, bb.liveout)
		printed = printbitset(printed, "avarinit", lv.vars, bb.avarinit)
		printed = printbitset(printed, "avarinitany", lv.vars, bb.avarinitany)
		printed = printbitset(printed, "avarinitall", lv.vars, bb.avarinitall)
		if printed != 0 {
			fmt.Printf("\n")
		}
	}

	fmt.Printf("\n")
}

// Dumps an array of bitmaps to a symbol as a sequence of uint32 values.  The
// first word dumped is the total number of bitmaps.  The second word is the
// length of the bitmaps.  All bitmaps are assumed to be of equal length.  The
// words that are followed are the raw bitmap words.  The arr argument is an
// array of Node*s.
func onebitwritesymbol(arr []Bvec, sym *Sym) {
	var i int
	var j int
	var word uint32

	n := len(arr)
	off := 0
	off += 4 // number of bitmaps, to fill in later
	bv := arr[0]
	off = duint32(sym, off, uint32(bv.n)) // number of bits in each bitmap
	for i = 0; i < n; i++ {
		// bitmap words
		bv = arr[i]

		if bv.b == nil {
			break
		}
		for j = 0; int32(j) < bv.n; j += 32 {
			word = bv.b[j/32]

			// Runtime reads the bitmaps as byte arrays. Oblige.
			off = duint8(sym, off, uint8(word))

			off = duint8(sym, off, uint8(word>>8))
			off = duint8(sym, off, uint8(word>>16))
			off = duint8(sym, off, uint8(word>>24))
		}
	}

	duint32(sym, 0, uint32(i)) // number of bitmaps
	ggloblsym(sym, int32(off), obj.RODATA)
}

func printprog(p *obj.Prog) {
	for p != nil {
		fmt.Printf("%v\n", p)
		p = p.Link
	}
}

// Entry pointer for liveness analysis.  Constructs a complete CFG, solves for
// the liveness of pointer variables in the function, and emits a runtime data
// structure read by the garbage collector.
func liveness(fn *Node, firstp *obj.Prog, argssym *Sym, livesym *Sym) {
	// Change name to dump debugging information only for a specific function.
	debugdelta := 0

	if Curfn.Func.Nname.Sym.Name == "!" {
		debugdelta = 2
	}

	debuglive += debugdelta
	if debuglive >= 3 {
		fmt.Printf("liveness: %s\n", Curfn.Func.Nname.Sym.Name)
		printprog(firstp)
	}

	checkptxt(fn, firstp)

	// Construct the global liveness state.
	cfg := newcfg(firstp)

	if debuglive >= 3 {
		printcfg([]*BasicBlock(cfg))
	}
	vars := getvariables(fn)
	lv := newliveness(fn, firstp, cfg, vars)

	// Run the dataflow framework.
	livenessprologue(lv)

	if debuglive >= 3 {
		livenessprintcfg(lv)
	}
	livenesssolve(lv)
	if debuglive >= 3 {
		livenessprintcfg(lv)
	}
	livenessepilogue(lv)
	if debuglive >= 3 {
		livenessprintcfg(lv)
	}
	livenesscompact(lv)

	if debuglive >= 2 {
		livenessprintdebug(lv)
	}

	// Emit the live pointer map data structures
	onebitwritesymbol(lv.livepointers, livesym)

	onebitwritesymbol(lv.argslivepointers, argssym)

	// Free everything.
	for l := fn.Func.Dcl; l != nil; l = l.Next {
		if l.N != nil {
			l.N.SetOpt(nil)
		}
	}
	freeliveness(lv)

	freecfg([]*BasicBlock(cfg))

	debuglive -= debugdelta
}
