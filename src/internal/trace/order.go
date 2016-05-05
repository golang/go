// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"fmt"
	"sort"
)

type eventBatch struct {
	events   []*Event
	selected bool
}

type orderEvent struct {
	ev    *Event
	batch int
	g     uint64
	init  gState
	next  gState
}

type gStatus int

type gState struct {
	seq    uint64
	status gStatus
}

const (
	gDead gStatus = iota
	gRunnable
	gRunning
	gWaiting

	unordered = ^uint64(0)
	garbage   = ^uint64(0) - 1
	noseq     = ^uint64(0)
	seqinc    = ^uint64(0) - 1
)

// order1007 merges a set of per-P event batches into a single, consistent stream.
// The high level idea is as follows. Events within an individual batch are in
// correct order, because they are emitted by a single P. So we need to produce
// a correct interleaving of the batches. To do this we take first unmerged event
// from each batch (frontier). Then choose subset that is "ready" to be merged,
// that is, events for which all dependencies are already merged. Then we choose
// event with the lowest timestamp from the subset, merge it and repeat.
// This approach ensures that we form a consistent stream even if timestamps are
// incorrect (condition observed on some machines).
func order1007(m map[int][]*Event) (events []*Event, err error) {
	pending := 0
	var batches []*eventBatch
	for _, v := range m {
		pending += len(v)
		batches = append(batches, &eventBatch{v, false})
	}
	gs := make(map[uint64]gState)
	var frontier []orderEvent
	for ; pending != 0; pending-- {
		for i, b := range batches {
			if b.selected || len(b.events) == 0 {
				continue
			}
			ev := b.events[0]
			g, init, next := stateTransition(ev)
			if !transitionReady(g, gs[g], init) {
				continue
			}
			frontier = append(frontier, orderEvent{ev, i, g, init, next})
			b.events = b.events[1:]
			b.selected = true
			// Get rid of "Local" events, they are intended merely for ordering.
			switch ev.Type {
			case EvGoStartLocal:
				ev.Type = EvGoStart
			case EvGoUnblockLocal:
				ev.Type = EvGoUnblock
			case EvGoSysExitLocal:
				ev.Type = EvGoSysExit
			}
		}
		if len(frontier) == 0 {
			return nil, fmt.Errorf("no consistent ordering of events possible")
		}
		sort.Sort(orderEventList(frontier))
		f := frontier[0]
		frontier[0] = frontier[len(frontier)-1]
		frontier = frontier[:len(frontier)-1]
		events = append(events, f.ev)
		transition(gs, f.g, f.init, f.next)
		if !batches[f.batch].selected {
			panic("frontier batch is not selected")
		}
		batches[f.batch].selected = false
	}

	// At this point we have a consistent stream of events.
	// Make sure time stamps respect the ordering.
	// The tests will skip (not fail) the test case if they see this error.
	if !sort.IsSorted(eventList(events)) {
		return nil, ErrTimeOrder
	}

	// The last part is giving correct timestamps to EvGoSysExit events.
	// The problem with EvGoSysExit is that actual syscall exit timestamp (ev.Args[2])
	// is potentially acquired long before event emission. So far we've used
	// timestamp of event emission (ev.Ts).
	// We could not set ev.Ts = ev.Args[2] earlier, because it would produce
	// seemingly broken timestamps (misplaced event).
	// We also can't simply update the timestamp and resort events, because
	// if timestamps are broken we will misplace the event and later report
	// logically broken trace (instead of reporting broken timestamps).
	lastSysBlock := make(map[uint64]int64)
	for _, ev := range events {
		switch ev.Type {
		case EvGoSysBlock, EvGoInSyscall:
			lastSysBlock[ev.G] = ev.Ts
		case EvGoSysExit:
			ts := int64(ev.Args[2])
			if ts == 0 {
				continue
			}
			block := lastSysBlock[ev.G]
			if block == 0 {
				return nil, fmt.Errorf("stray syscall exit")
			}
			if ts < block {
				return nil, ErrTimeOrder
			}
			ev.Ts = ts
		}
	}
	sort.Stable(eventList(events))

	return
}

// stateTransition returns goroutine state (sequence and status) when the event
// becomes ready for merging (init) and the goroutine state after the event (next).
func stateTransition(ev *Event) (g uint64, init, next gState) {
	switch ev.Type {
	case EvGoCreate:
		g = ev.Args[0]
		init = gState{0, gDead}
		next = gState{1, gRunnable}
	case EvGoWaiting, EvGoInSyscall:
		g = ev.G
		init = gState{1, gRunnable}
		next = gState{2, gWaiting}
	case EvGoStart:
		g = ev.G
		init = gState{ev.Args[1], gRunnable}
		next = gState{ev.Args[1] + 1, gRunning}
	case EvGoStartLocal:
		// noseq means that this event is ready for merging as soon as
		// frontier reaches it (EvGoStartLocal is emitted on the same P
		// as the corresponding EvGoCreate/EvGoUnblock, and thus the latter
		// is already merged).
		// seqinc is a stub for cases when event increments g sequence,
		// but since we don't know current seq we also don't know next seq.
		g = ev.G
		init = gState{noseq, gRunnable}
		next = gState{seqinc, gRunning}
	case EvGoBlock, EvGoBlockSend, EvGoBlockRecv, EvGoBlockSelect,
		EvGoBlockSync, EvGoBlockCond, EvGoBlockNet, EvGoSleep, EvGoSysBlock:
		g = ev.G
		init = gState{noseq, gRunning}
		next = gState{noseq, gWaiting}
	case EvGoSched, EvGoPreempt:
		g = ev.G
		init = gState{noseq, gRunning}
		next = gState{noseq, gRunnable}
	case EvGoUnblock, EvGoSysExit:
		g = ev.Args[0]
		init = gState{ev.Args[1], gWaiting}
		next = gState{ev.Args[1] + 1, gRunnable}
	case EvGoUnblockLocal, EvGoSysExitLocal:
		g = ev.Args[0]
		init = gState{noseq, gWaiting}
		next = gState{seqinc, gRunnable}
	case EvGCStart:
		g = garbage
		init = gState{ev.Args[0], gDead}
		next = gState{ev.Args[0] + 1, gDead}
	default:
		// no ordering requirements
		g = unordered
	}
	return
}

func transitionReady(g uint64, curr, init gState) bool {
	return g == unordered || (init.seq == noseq || init.seq == curr.seq) && init.status == curr.status
}

func transition(gs map[uint64]gState, g uint64, init, next gState) {
	if g == unordered {
		return
	}
	curr := gs[g]
	if !transitionReady(g, curr, init) {
		panic("event sequences are broken")
	}
	switch next.seq {
	case noseq:
		next.seq = curr.seq
	case seqinc:
		next.seq = curr.seq + 1
	}
	gs[g] = next
}

// order1005 merges a set of per-P event batches into a single, consistent stream.
func order1005(m map[int][]*Event) (events []*Event, err error) {
	for _, batch := range m {
		events = append(events, batch...)
	}
	for _, ev := range events {
		if ev.Type == EvGoSysExit {
			// EvGoSysExit emission is delayed until the thread has a P.
			// Give it the real sequence number and time stamp.
			ev.seq = int64(ev.Args[1])
			if ev.Args[2] != 0 {
				ev.Ts = int64(ev.Args[2])
			}
		}
	}
	sort.Sort(eventSeqList(events))
	if !sort.IsSorted(eventList(events)) {
		return nil, ErrTimeOrder
	}
	return
}

type orderEventList []orderEvent

func (l orderEventList) Len() int {
	return len(l)
}

func (l orderEventList) Less(i, j int) bool {
	return l[i].ev.Ts < l[j].ev.Ts
}

func (l orderEventList) Swap(i, j int) {
	l[i], l[j] = l[j], l[i]
}

type eventList []*Event

func (l eventList) Len() int {
	return len(l)
}

func (l eventList) Less(i, j int) bool {
	return l[i].Ts < l[j].Ts
}

func (l eventList) Swap(i, j int) {
	l[i], l[j] = l[j], l[i]
}

type eventSeqList []*Event

func (l eventSeqList) Len() int {
	return len(l)
}

func (l eventSeqList) Less(i, j int) bool {
	return l[i].seq < l[j].seq
}

func (l eventSeqList) Swap(i, j int) {
	l[i], l[j] = l[j], l[i]
}
