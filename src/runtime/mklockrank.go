// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

// mklockrank records the static rank graph of the locks in the
// runtime and generates the rank checking structures in lockrank.go.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/format"
	"internal/dag"
	"io"
	"log"
	"os"
	"strings"
)

// ranks describes the lock rank graph. See "go doc internal/dag" for
// the syntax.
//
// "a < b" means a must be acquired before b if both are held
// (or, if b is held, a cannot be acquired).
//
// "NONE < a" means no locks may be held when a is acquired.
//
// If a lock is not given a rank, then it is assumed to be a leaf
// lock, which means no other lock can be acquired while it is held.
// Therefore, leaf locks do not need to be given an explicit rank.
//
// Ranks in all caps are pseudo-nodes that help define order, but do
// not actually define a rank.
//
// TODO: It's often hard to correlate rank names to locks. Change
// these to be more consistent with the locks they label.
const ranks = `
# Sysmon
NONE
< sysmon
< scavenge, forcegc, updateGOMAXPROCS;

# Defer
NONE < defer;

# GC
NONE <
  sweepWaiters,
  assistQueue,
  strongFromWeakQueue,
  cleanupQueue,
  sweep;

# Test only
NONE < testR, testW;

NONE < timerSend;

# Scheduler, timers, netpoll
NONE < allocmW, execW, cpuprof, pollCache, pollDesc, wakeableSleep;
scavenge, sweep, testR, wakeableSleep, timerSend < hchan;
assistQueue,
  cleanupQueue,
  cpuprof,
  forcegc,
  updateGOMAXPROCS,
  hchan,
  pollDesc, # pollDesc can interact with timers, which can lock sched.
  scavenge,
  strongFromWeakQueue,
  sweep,
  sweepWaiters,
  testR,
  wakeableSleep
# Above SCHED are things that can call into the scheduler.
< SCHED
# Below SCHED is the scheduler implementation.
< allocmR,
  execR;
allocmR, execR, hchan < sched;
sched < allg, allp;

# Channels
NONE < notifyList;
hchan, notifyList < sudog;

hchan, pollDesc, wakeableSleep < timers;
timers, timerSend < timer < netpollInit;

# Semaphores
NONE < root;

# Itabs
NONE
< itab
< reflectOffs;

# Synctest
hchan, root, timers, timer, notifyList, reflectOffs < synctest;

# User arena state
NONE < userArenaState;

# Tracing without a P uses a global trace buffer.
scavenge
# Above TRACEGLOBAL can emit a trace event without a P.
< TRACEGLOBAL
# Below TRACEGLOBAL manages the global tracing buffer.
# Note that traceBuf eventually chains to MALLOC, but we never get that far
# in the situation where there's no P.
< traceBuf;
# Starting/stopping tracing traces strings.
traceBuf < traceStrings;

# Malloc
allg,
  allocmR,
  allp, # procresize
  execR, # May grow stack
  execW, # May allocate after BeforeFork
  hchan,
  notifyList,
  reflectOffs,
  timer,
  traceStrings,
  userArenaState
# Above MALLOC are things that can allocate memory.
< MALLOC
# Below MALLOC is the malloc implementation.
< fin,
  spanSetSpine,
  mspanSpecial,
  traceTypeTab,
  MPROF;

# We can acquire gcBitsArenas for pinner bits, and
# it's guarded by mspanSpecial.
MALLOC, mspanSpecial < gcBitsArenas;

# Memory profiling
MPROF < profInsert, profBlock, profMemActive;
profMemActive < profMemFuture;

# Stack allocation and copying
gcBitsArenas,
  netpollInit,
  profBlock,
  profInsert,
  profMemFuture,
  spanSetSpine,
  synctest,
  fin,
  root
# Anything that can grow the stack can acquire STACKGROW.
# (Most higher layers imply STACKGROW, like MALLOC.)
< STACKGROW
# Below STACKGROW is the stack allocator/copying implementation.
< gscan;
gscan < stackpool;
gscan < stackLarge;
# Generally, hchan must be acquired before gscan. But in one case,
# where we suspend a G and then shrink its stack, syncadjustsudogs
# can acquire hchan locks while holding gscan. To allow this case,
# we use hchanLeaf instead of hchan.
gscan < hchanLeaf;

# Write barrier
defer,
  gscan,
  mspanSpecial,
  pollCache,
  sudog,
  timer
# Anything that can have write barriers can acquire WB.
# Above WB, we can have write barriers.
< WB
# Below WB is the write barrier implementation.
< wbufSpans;

# Span allocator
stackLarge,
  stackpool,
  wbufSpans
# Above mheap is anything that can call the span allocator.
< mheap;
# Below mheap is the span allocator implementation.
#
# Specials: we're allowed to allocate a special while holding
# an mspanSpecial lock, and they're part of the malloc implementation.
# Pinner bits might be freed by the span allocator.
mheap, mspanSpecial < mheapSpecial;
mheap, mheapSpecial < globalAlloc;

# Execution tracer events (with a P)
hchan,
  mheap,
  root,
  sched,
  traceStrings,
  notifyList,
  fin
# Above TRACE is anything that can create a trace event
< TRACE
< trace
< traceStackTab;

# panic is handled specially. It is implicitly below all other locks.
NONE < panic;
# deadlock is not acquired while holding panic, but it also needs to be
# below all other locks.
panic < deadlock;
# raceFini is only held while exiting.
panic < raceFini;

# RWMutex internal read lock

allocmR,
  allocmW
< allocmRInternal;

execR,
  execW
< execRInternal;

testR,
  testW
< testRInternal;
`

// cyclicRanks lists lock ranks that allow multiple locks of the same
// rank to be acquired simultaneously. The runtime enforces ordering
// within these ranks using a separate mechanism.
var cyclicRanks = map[string]bool{
	// Multiple timers are locked simultaneously in destroy().
	"timers": true,
	// Multiple hchans are acquired in hchan.sortkey() order in
	// select.
	"hchan": true,
	// Multiple hchanLeafs are acquired in hchan.sortkey() order in
	// syncadjustsudogs().
	"hchanLeaf": true,
	// The point of the deadlock lock is to deadlock.
	"deadlock": true,
}

func main() {
	flagO := flag.String("o", "", "write to `file` instead of stdout")
	flagDot := flag.Bool("dot", false, "emit graphviz output instead of Go")
	flag.Parse()
	if flag.NArg() != 0 {
		fmt.Fprintf(os.Stderr, "too many arguments")
		os.Exit(2)
	}

	g, err := dag.Parse(ranks)
	if err != nil {
		log.Fatal(err)
	}

	var out []byte
	if *flagDot {
		var b bytes.Buffer
		g.TransitiveReduction()
		// Add cyclic edges for visualization.
		for k := range cyclicRanks {
			g.AddEdge(k, k)
		}
		// Reverse the graph. It's much easier to read this as
		// a "<" partial order than a ">" partial order. This
		// ways, locks are acquired from the top going down
		// and time moves forward over the edges instead of
		// backward.
		g.Transpose()
		generateDot(&b, g)
		out = b.Bytes()
	} else {
		var b bytes.Buffer
		generateGo(&b, g)
		out, err = format.Source(b.Bytes())
		if err != nil {
			log.Fatal(err)
		}
	}

	if *flagO != "" {
		err = os.WriteFile(*flagO, out, 0666)
	} else {
		_, err = os.Stdout.Write(out)
	}
	if err != nil {
		log.Fatal(err)
	}
}

func generateGo(w io.Writer, g *dag.Graph) {
	fmt.Fprintf(w, `// Code generated by mklockrank.go; DO NOT EDIT.

package runtime

type lockRank int

`)

	// Create numeric ranks.
	topo := g.Topo()
	for i, j := 0, len(topo)-1; i < j; i, j = i+1, j-1 {
		topo[i], topo[j] = topo[j], topo[i]
	}
	fmt.Fprintf(w, `
// Constants representing the ranks of all non-leaf runtime locks, in rank order.
// Locks with lower rank must be taken before locks with higher rank,
// in addition to satisfying the partial order in lockPartialOrder.
// A few ranks allow self-cycles, which are specified in lockPartialOrder.
const (
	lockRankUnknown lockRank = iota

`)
	for _, rank := range topo {
		if isPseudo(rank) {
			fmt.Fprintf(w, "\t// %s\n", rank)
		} else {
			fmt.Fprintf(w, "\t%s\n", cname(rank))
		}
	}
	fmt.Fprintf(w, `)

// lockRankLeafRank is the rank of lock that does not have a declared rank,
// and hence is a leaf lock.
const lockRankLeafRank lockRank = 1000
`)

	// Create string table.
	fmt.Fprintf(w, `
// lockNames gives the names associated with each of the above ranks.
var lockNames = []string{
`)
	for _, rank := range topo {
		if !isPseudo(rank) {
			fmt.Fprintf(w, "\t%s: %q,\n", cname(rank), rank)
		}
	}
	fmt.Fprintf(w, `}

func (rank lockRank) String() string {
	if rank == 0 {
		return "UNKNOWN"
	}
	if rank == lockRankLeafRank {
		return "LEAF"
	}
	if rank < 0 || int(rank) >= len(lockNames) {
		return "BAD RANK"
	}
	return lockNames[rank]
}
`)

	// Create partial order structure.
	fmt.Fprintf(w, `
// lockPartialOrder is the transitive closure of the lock rank graph.
// An entry for rank X lists all of the ranks that can already be held
// when rank X is acquired.
//
// Lock ranks that allow self-cycles list themselves.
var lockPartialOrder [][]lockRank = [][]lockRank{
`)
	for _, rank := range topo {
		if isPseudo(rank) {
			continue
		}
		list := []string{}
		for _, before := range g.Edges(rank) {
			if !isPseudo(before) {
				list = append(list, cname(before))
			}
		}
		if cyclicRanks[rank] {
			list = append(list, cname(rank))
		}

		fmt.Fprintf(w, "\t%s: {%s},\n", cname(rank), strings.Join(list, ", "))
	}
	fmt.Fprintf(w, "}\n")
}

// cname returns the Go const name for the given lock rank label.
func cname(label string) string {
	return "lockRank" + strings.ToUpper(label[:1]) + label[1:]
}

func isPseudo(label string) bool {
	return strings.ToUpper(label) == label
}

// generateDot emits a Graphviz dot representation of g to w.
func generateDot(w io.Writer, g *dag.Graph) {
	fmt.Fprintf(w, "digraph g {\n")

	// Define all nodes.
	for _, node := range g.Nodes {
		fmt.Fprintf(w, "%q;\n", node)
	}

	// Create edges.
	for _, node := range g.Nodes {
		for _, to := range g.Edges(node) {
			fmt.Fprintf(w, "%q -> %q;\n", node, to)
		}
	}

	fmt.Fprintf(w, "}\n")
}
