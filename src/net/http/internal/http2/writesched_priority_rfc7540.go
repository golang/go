// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"fmt"
	"math"
	"sort"
)

// RFC 7540, Section 5.3.5: the default weight is 16.
const priorityDefaultWeightRFC7540 = 15 // 16 = 15 + 1

// PriorityWriteSchedulerConfig configures a priorityWriteScheduler.
type PriorityWriteSchedulerConfig struct {
	// MaxClosedNodesInTree controls the maximum number of closed streams to
	// retain in the priority tree. Setting this to zero saves a small amount
	// of memory at the cost of performance.
	//
	// See RFC 7540, Section 5.3.4:
	//   "It is possible for a stream to become closed while prioritization
	//   information ... is in transit. ... This potentially creates suboptimal
	//   prioritization, since the stream could be given a priority that is
	//   different from what is intended. To avoid these problems, an endpoint
	//   SHOULD retain stream prioritization state for a period after streams
	//   become closed. The longer state is retained, the lower the chance that
	//   streams are assigned incorrect or default priority values."
	MaxClosedNodesInTree int

	// MaxIdleNodesInTree controls the maximum number of idle streams to
	// retain in the priority tree. Setting this to zero saves a small amount
	// of memory at the cost of performance.
	//
	// See RFC 7540, Section 5.3.4:
	//   Similarly, streams that are in the "idle" state can be assigned
	//   priority or become a parent of other streams. This allows for the
	//   creation of a grouping node in the dependency tree, which enables
	//   more flexible expressions of priority. Idle streams begin with a
	//   default priority (Section 5.3.5).
	MaxIdleNodesInTree int

	// ThrottleOutOfOrderWrites enables write throttling to help ensure that
	// data is delivered in priority order. This works around a race where
	// stream B depends on stream A and both streams are about to call Write
	// to queue DATA frames. If B wins the race, a naive scheduler would eagerly
	// write as much data from B as possible, but this is suboptimal because A
	// is a higher-priority stream. With throttling enabled, we write a small
	// amount of data from B to minimize the amount of bandwidth that B can
	// steal from A.
	ThrottleOutOfOrderWrites bool
}

// NewPriorityWriteScheduler constructs a WriteScheduler that schedules
// frames by following HTTP/2 priorities as described in RFC 7540 Section 5.3.
// If cfg is nil, default options are used.
func NewPriorityWriteScheduler(cfg *PriorityWriteSchedulerConfig) WriteScheduler {
	return newPriorityWriteSchedulerRFC7540(cfg)
}

func newPriorityWriteSchedulerRFC7540(cfg *PriorityWriteSchedulerConfig) WriteScheduler {
	if cfg == nil {
		// For justification of these defaults, see:
		// https://docs.google.com/document/d/1oLhNg1skaWD4_DtaoCxdSRN5erEXrH-KnLrMwEpOtFY
		cfg = &PriorityWriteSchedulerConfig{
			MaxClosedNodesInTree:     10,
			MaxIdleNodesInTree:       10,
			ThrottleOutOfOrderWrites: false,
		}
	}

	ws := &priorityWriteSchedulerRFC7540{
		nodes:                make(map[uint32]*priorityNodeRFC7540),
		maxClosedNodesInTree: cfg.MaxClosedNodesInTree,
		maxIdleNodesInTree:   cfg.MaxIdleNodesInTree,
		enableWriteThrottle:  cfg.ThrottleOutOfOrderWrites,
	}
	ws.nodes[0] = &ws.root
	if cfg.ThrottleOutOfOrderWrites {
		ws.writeThrottleLimit = 1024
	} else {
		ws.writeThrottleLimit = math.MaxInt32
	}
	return ws
}

type priorityNodeStateRFC7540 int

const (
	priorityNodeOpenRFC7540 priorityNodeStateRFC7540 = iota
	priorityNodeClosedRFC7540
	priorityNodeIdleRFC7540
)

// priorityNodeRFC7540 is a node in an HTTP/2 priority tree.
// Each node is associated with a single stream ID.
// See RFC 7540, Section 5.3.
type priorityNodeRFC7540 struct {
	q            writeQueue               // queue of pending frames to write
	id           uint32                   // id of the stream, or 0 for the root of the tree
	weight       uint8                    // the actual weight is weight+1, so the value is in [1,256]
	state        priorityNodeStateRFC7540 // open | closed | idle
	bytes        int64                    // number of bytes written by this node, or 0 if closed
	subtreeBytes int64                    // sum(node.bytes) of all nodes in this subtree

	// These links form the priority tree.
	parent     *priorityNodeRFC7540
	kids       *priorityNodeRFC7540 // start of the kids list
	prev, next *priorityNodeRFC7540 // doubly-linked list of siblings
}

func (n *priorityNodeRFC7540) setParent(parent *priorityNodeRFC7540) {
	if n == parent {
		panic("setParent to self")
	}
	if n.parent == parent {
		return
	}
	// Unlink from current parent.
	if parent := n.parent; parent != nil {
		if n.prev == nil {
			parent.kids = n.next
		} else {
			n.prev.next = n.next
		}
		if n.next != nil {
			n.next.prev = n.prev
		}
	}
	// Link to new parent.
	// If parent=nil, remove n from the tree.
	// Always insert at the head of parent.kids (this is assumed by walkReadyInOrder).
	n.parent = parent
	if parent == nil {
		n.next = nil
		n.prev = nil
	} else {
		n.next = parent.kids
		n.prev = nil
		if n.next != nil {
			n.next.prev = n
		}
		parent.kids = n
	}
}

func (n *priorityNodeRFC7540) addBytes(b int64) {
	n.bytes += b
	for ; n != nil; n = n.parent {
		n.subtreeBytes += b
	}
}

// walkReadyInOrder iterates over the tree in priority order, calling f for each node
// with a non-empty write queue. When f returns true, this function returns true and the
// walk halts. tmp is used as scratch space for sorting.
//
// f(n, openParent) takes two arguments: the node to visit, n, and a bool that is true
// if any ancestor p of n is still open (ignoring the root node).
func (n *priorityNodeRFC7540) walkReadyInOrder(openParent bool, tmp *[]*priorityNodeRFC7540, f func(*priorityNodeRFC7540, bool) bool) bool {
	if !n.q.empty() && f(n, openParent) {
		return true
	}
	if n.kids == nil {
		return false
	}

	// Don't consider the root "open" when updating openParent since
	// we can't send data frames on the root stream (only control frames).
	if n.id != 0 {
		openParent = openParent || (n.state == priorityNodeOpenRFC7540)
	}

	// Common case: only one kid or all kids have the same weight.
	// Some clients don't use weights; other clients (like web browsers)
	// use mostly-linear priority trees.
	w := n.kids.weight
	needSort := false
	for k := n.kids.next; k != nil; k = k.next {
		if k.weight != w {
			needSort = true
			break
		}
	}
	if !needSort {
		for k := n.kids; k != nil; k = k.next {
			if k.walkReadyInOrder(openParent, tmp, f) {
				return true
			}
		}
		return false
	}

	// Uncommon case: sort the child nodes. We remove the kids from the parent,
	// then re-insert after sorting so we can reuse tmp for future sort calls.
	*tmp = (*tmp)[:0]
	for n.kids != nil {
		*tmp = append(*tmp, n.kids)
		n.kids.setParent(nil)
	}
	sort.Sort(sortPriorityNodeSiblingsRFC7540(*tmp))
	for i := len(*tmp) - 1; i >= 0; i-- {
		(*tmp)[i].setParent(n) // setParent inserts at the head of n.kids
	}
	for k := n.kids; k != nil; k = k.next {
		if k.walkReadyInOrder(openParent, tmp, f) {
			return true
		}
	}
	return false
}

type sortPriorityNodeSiblingsRFC7540 []*priorityNodeRFC7540

func (z sortPriorityNodeSiblingsRFC7540) Len() int      { return len(z) }
func (z sortPriorityNodeSiblingsRFC7540) Swap(i, k int) { z[i], z[k] = z[k], z[i] }
func (z sortPriorityNodeSiblingsRFC7540) Less(i, k int) bool {
	// Prefer the subtree that has sent fewer bytes relative to its weight.
	// See sections 5.3.2 and 5.3.4.
	wi, bi := float64(z[i].weight)+1, float64(z[i].subtreeBytes)
	wk, bk := float64(z[k].weight)+1, float64(z[k].subtreeBytes)
	if bi == 0 && bk == 0 {
		return wi >= wk
	}
	if bk == 0 {
		return false
	}
	return bi/bk <= wi/wk
}

type priorityWriteSchedulerRFC7540 struct {
	// root is the root of the priority tree, where root.id = 0.
	// The root queues control frames that are not associated with any stream.
	root priorityNodeRFC7540

	// nodes maps stream ids to priority tree nodes.
	nodes map[uint32]*priorityNodeRFC7540

	// maxID is the maximum stream id in nodes.
	maxID uint32

	// lists of nodes that have been closed or are idle, but are kept in
	// the tree for improved prioritization. When the lengths exceed either
	// maxClosedNodesInTree or maxIdleNodesInTree, old nodes are discarded.
	closedNodes, idleNodes []*priorityNodeRFC7540

	// From the config.
	maxClosedNodesInTree int
	maxIdleNodesInTree   int
	writeThrottleLimit   int32
	enableWriteThrottle  bool

	// tmp is scratch space for priorityNode.walkReadyInOrder to reduce allocations.
	tmp []*priorityNodeRFC7540

	// pool of empty queues for reuse.
	queuePool writeQueuePool
}

func (ws *priorityWriteSchedulerRFC7540) OpenStream(streamID uint32, options OpenStreamOptions) {
	// The stream may be currently idle but cannot be opened or closed.
	if curr := ws.nodes[streamID]; curr != nil {
		if curr.state != priorityNodeIdleRFC7540 {
			panic(fmt.Sprintf("stream %d already opened", streamID))
		}
		curr.state = priorityNodeOpenRFC7540
		return
	}

	// RFC 7540, Section 5.3.5:
	//  "All streams are initially assigned a non-exclusive dependency on stream 0x0.
	//  Pushed streams initially depend on their associated stream. In both cases,
	//  streams are assigned a default weight of 16."
	parent := ws.nodes[options.PusherID]
	if parent == nil {
		parent = &ws.root
	}
	n := &priorityNodeRFC7540{
		q:      *ws.queuePool.get(),
		id:     streamID,
		weight: priorityDefaultWeightRFC7540,
		state:  priorityNodeOpenRFC7540,
	}
	n.setParent(parent)
	ws.nodes[streamID] = n
	if streamID > ws.maxID {
		ws.maxID = streamID
	}
}

func (ws *priorityWriteSchedulerRFC7540) CloseStream(streamID uint32) {
	if streamID == 0 {
		panic("violation of WriteScheduler interface: cannot close stream 0")
	}
	if ws.nodes[streamID] == nil {
		panic(fmt.Sprintf("violation of WriteScheduler interface: unknown stream %d", streamID))
	}
	if ws.nodes[streamID].state != priorityNodeOpenRFC7540 {
		panic(fmt.Sprintf("violation of WriteScheduler interface: stream %d already closed", streamID))
	}

	n := ws.nodes[streamID]
	n.state = priorityNodeClosedRFC7540
	n.addBytes(-n.bytes)

	q := n.q
	ws.queuePool.put(&q)
	if ws.maxClosedNodesInTree > 0 {
		ws.addClosedOrIdleNode(&ws.closedNodes, ws.maxClosedNodesInTree, n)
	} else {
		ws.removeNode(n)
	}
}

func (ws *priorityWriteSchedulerRFC7540) AdjustStream(streamID uint32, priority PriorityParam) {
	if streamID == 0 {
		panic("adjustPriority on root")
	}

	// If streamID does not exist, there are two cases:
	// - A closed stream that has been removed (this will have ID <= maxID)
	// - An idle stream that is being used for "grouping" (this will have ID > maxID)
	n := ws.nodes[streamID]
	if n == nil {
		if streamID <= ws.maxID || ws.maxIdleNodesInTree == 0 {
			return
		}
		ws.maxID = streamID
		n = &priorityNodeRFC7540{
			q:      *ws.queuePool.get(),
			id:     streamID,
			weight: priorityDefaultWeightRFC7540,
			state:  priorityNodeIdleRFC7540,
		}
		n.setParent(&ws.root)
		ws.nodes[streamID] = n
		ws.addClosedOrIdleNode(&ws.idleNodes, ws.maxIdleNodesInTree, n)
	}

	// Section 5.3.1: A dependency on a stream that is not currently in the tree
	// results in that stream being given a default priority (Section 5.3.5).
	parent := ws.nodes[priority.StreamDep]
	if parent == nil {
		n.setParent(&ws.root)
		n.weight = priorityDefaultWeightRFC7540
		return
	}

	// Ignore if the client tries to make a node its own parent.
	if n == parent {
		return
	}

	// Section 5.3.3:
	//   "If a stream is made dependent on one of its own dependencies, the
	//   formerly dependent stream is first moved to be dependent on the
	//   reprioritized stream's previous parent. The moved dependency retains
	//   its weight."
	//
	// That is: if parent depends on n, move parent to depend on n.parent.
	for x := parent.parent; x != nil; x = x.parent {
		if x == n {
			parent.setParent(n.parent)
			break
		}
	}

	// Section 5.3.3: The exclusive flag causes the stream to become the sole
	// dependency of its parent stream, causing other dependencies to become
	// dependent on the exclusive stream.
	if priority.Exclusive {
		k := parent.kids
		for k != nil {
			next := k.next
			if k != n {
				k.setParent(n)
			}
			k = next
		}
	}

	n.setParent(parent)
	n.weight = priority.Weight
}

func (ws *priorityWriteSchedulerRFC7540) Push(wr FrameWriteRequest) {
	var n *priorityNodeRFC7540
	if wr.isControl() {
		n = &ws.root
	} else {
		id := wr.StreamID()
		n = ws.nodes[id]
		if n == nil {
			// id is an idle or closed stream. wr should not be a HEADERS or
			// DATA frame. In other case, we push wr onto the root, rather
			// than creating a new priorityNode.
			if wr.DataSize() > 0 {
				panic("add DATA on non-open stream")
			}
			n = &ws.root
		}
	}
	n.q.push(wr)
}

func (ws *priorityWriteSchedulerRFC7540) Pop() (wr FrameWriteRequest, ok bool) {
	ws.root.walkReadyInOrder(false, &ws.tmp, func(n *priorityNodeRFC7540, openParent bool) bool {
		limit := int32(math.MaxInt32)
		if openParent {
			limit = ws.writeThrottleLimit
		}
		wr, ok = n.q.consume(limit)
		if !ok {
			return false
		}
		n.addBytes(int64(wr.DataSize()))
		// If B depends on A and B continuously has data available but A
		// does not, gradually increase the throttling limit to allow B to
		// steal more and more bandwidth from A.
		if openParent {
			ws.writeThrottleLimit += 1024
			if ws.writeThrottleLimit < 0 {
				ws.writeThrottleLimit = math.MaxInt32
			}
		} else if ws.enableWriteThrottle {
			ws.writeThrottleLimit = 1024
		}
		return true
	})
	return wr, ok
}

func (ws *priorityWriteSchedulerRFC7540) addClosedOrIdleNode(list *[]*priorityNodeRFC7540, maxSize int, n *priorityNodeRFC7540) {
	if maxSize == 0 {
		return
	}
	if len(*list) == maxSize {
		// Remove the oldest node, then shift left.
		ws.removeNode((*list)[0])
		x := (*list)[1:]
		copy(*list, x)
		*list = (*list)[:len(x)]
	}
	*list = append(*list, n)
}

func (ws *priorityWriteSchedulerRFC7540) removeNode(n *priorityNodeRFC7540) {
	for n.kids != nil {
		n.kids.setParent(n.parent)
	}
	n.setParent(nil)
	delete(ws.nodes, n.id)
}
