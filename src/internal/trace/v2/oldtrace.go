// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements conversion from old (Go 1.11–Go 1.21) traces to the Go
// 1.22 format.
//
// Most events have direct equivalents in 1.22, at worst requiring arguments to
// be reordered. Some events, such as GoWaiting need to look ahead for follow-up
// events to determine the correct translation. GoSyscall, which is an
// instantaneous event, gets turned into a 1 ns long pair of
// GoSyscallStart+GoSyscallEnd, unless we observe a GoSysBlock, in which case we
// emit a GoSyscallStart+GoSyscallEndBlocked pair with the correct duration
// (i.e. starting at the original GoSyscall).
//
// The resulting trace treats the old trace as a single, large generation,
// sharing a single evTable for all events.
//
// We use a new (compared to what was used for 'go tool trace' in earlier
// versions of Go) parser for old traces that is optimized for speed, low memory
// usage, and minimal GC pressure. It allocates events in batches so that even
// though we have to load the entire trace into memory, the conversion process
// shouldn't result in a doubling of memory usage, even if all converted events
// are kept alive, as we free batches once we're done with them.
//
// The conversion process is lossless.

package trace

import (
	"errors"
	"fmt"
	"internal/trace/v2/event"
	"internal/trace/v2/event/go122"
	"internal/trace/v2/internal/oldtrace"
	"io"
)

type oldTraceConverter struct {
	trace          oldtrace.Trace
	evt            *evTable
	preInit        bool
	createdPreInit map[GoID]struct{}
	events         oldtrace.Events
	extra          []Event
	extraArr       [3]Event
	tasks          map[TaskID]taskState
	seenProcs      map[ProcID]struct{}
	lastTs         Time
	procMs         map[ProcID]ThreadID
	lastStwReason  uint64

	inlineToStringID  []uint64
	builtinToStringID []uint64
}

const (
	// Block reasons
	sForever = iota
	sPreempted
	sGosched
	sSleep
	sChanSend
	sChanRecv
	sNetwork
	sSync
	sSyncCond
	sSelect
	sEmpty
	sMarkAssistWait

	// STW kinds
	sSTWUnknown
	sSTWGCMarkTermination
	sSTWGCSweepTermination
	sSTWWriteHeapDump
	sSTWGoroutineProfile
	sSTWGoroutineProfileCleanup
	sSTWAllGoroutinesStackTrace
	sSTWReadMemStats
	sSTWAllThreadsSyscall
	sSTWGOMAXPROCS
	sSTWStartTrace
	sSTWStopTrace
	sSTWCountPagesInUse
	sSTWReadMetricsSlow
	sSTWReadMemStatsSlow
	sSTWPageCachePagesLeaked
	sSTWResetDebugLog

	sLast
)

func (it *oldTraceConverter) init(pr oldtrace.Trace) error {
	it.trace = pr
	it.preInit = true
	it.createdPreInit = make(map[GoID]struct{})
	it.evt = &evTable{pcs: make(map[uint64]frame)}
	it.events = pr.Events
	it.extra = it.extraArr[:0]
	it.tasks = make(map[TaskID]taskState)
	it.seenProcs = make(map[ProcID]struct{})
	it.procMs = make(map[ProcID]ThreadID)
	it.lastTs = -1

	evt := it.evt

	// Convert from oldtracer's Strings map to our dataTable.
	var max uint64
	for id, s := range pr.Strings {
		evt.strings.insert(stringID(id), s)
		if id > max {
			max = id
		}
	}
	pr.Strings = nil

	// Add all strings used for UserLog. In the old trace format, these were
	// stored inline and didn't have IDs. We generate IDs for them.
	if max+uint64(len(pr.InlineStrings)) < max {
		return errors.New("trace contains too many strings")
	}
	var addErr error
	add := func(id stringID, s string) {
		if err := evt.strings.insert(id, s); err != nil && addErr == nil {
			addErr = err
		}
	}
	for id, s := range pr.InlineStrings {
		nid := max + 1 + uint64(id)
		it.inlineToStringID = append(it.inlineToStringID, nid)
		add(stringID(nid), s)
	}
	max += uint64(len(pr.InlineStrings))
	pr.InlineStrings = nil

	// Add strings that the converter emits explicitly.
	if max+uint64(sLast) < max {
		return errors.New("trace contains too many strings")
	}
	it.builtinToStringID = make([]uint64, sLast)
	addBuiltin := func(c int, s string) {
		nid := max + 1 + uint64(c)
		it.builtinToStringID[c] = nid
		add(stringID(nid), s)
	}
	addBuiltin(sForever, "forever")
	addBuiltin(sPreempted, "preempted")
	addBuiltin(sGosched, "runtime.Gosched")
	addBuiltin(sSleep, "sleep")
	addBuiltin(sChanSend, "chan send")
	addBuiltin(sChanRecv, "chan receive")
	addBuiltin(sNetwork, "network")
	addBuiltin(sSync, "sync")
	addBuiltin(sSyncCond, "sync.(*Cond).Wait")
	addBuiltin(sSelect, "select")
	addBuiltin(sEmpty, "")
	addBuiltin(sMarkAssistWait, "GC mark assist wait for work")
	addBuiltin(sSTWUnknown, "")
	addBuiltin(sSTWGCMarkTermination, "GC mark termination")
	addBuiltin(sSTWGCSweepTermination, "GC sweep termination")
	addBuiltin(sSTWWriteHeapDump, "write heap dump")
	addBuiltin(sSTWGoroutineProfile, "goroutine profile")
	addBuiltin(sSTWGoroutineProfileCleanup, "goroutine profile cleanup")
	addBuiltin(sSTWAllGoroutinesStackTrace, "all goroutine stack trace")
	addBuiltin(sSTWReadMemStats, "read mem stats")
	addBuiltin(sSTWAllThreadsSyscall, "AllThreadsSyscall")
	addBuiltin(sSTWGOMAXPROCS, "GOMAXPROCS")
	addBuiltin(sSTWStartTrace, "start trace")
	addBuiltin(sSTWStopTrace, "stop trace")
	addBuiltin(sSTWCountPagesInUse, "CountPagesInUse (test)")
	addBuiltin(sSTWReadMetricsSlow, "ReadMetricsSlow (test)")
	addBuiltin(sSTWReadMemStatsSlow, "ReadMemStatsSlow (test)")
	addBuiltin(sSTWPageCachePagesLeaked, "PageCachePagesLeaked (test)")
	addBuiltin(sSTWResetDebugLog, "ResetDebugLog (test)")

	if addErr != nil {
		// This should be impossible but let's be safe.
		return fmt.Errorf("couldn't add strings: %w", addErr)
	}

	it.evt.strings.compactify()

	// Convert stacks.
	for id, stk := range pr.Stacks {
		evt.stacks.insert(stackID(id), stack{pcs: stk})
	}

	// OPT(dh): if we could share the frame type between this package and
	// oldtrace we wouldn't have to copy the map.
	for pc, f := range pr.PCs {
		evt.pcs[pc] = frame{
			pc:     pc,
			funcID: stringID(f.Fn),
			fileID: stringID(f.File),
			line:   uint64(f.Line),
		}
	}
	pr.Stacks = nil
	pr.PCs = nil
	evt.stacks.compactify()
	return nil
}

// next returns the next event, io.EOF if there are no more events, or a
// descriptive error for invalid events.
func (it *oldTraceConverter) next() (Event, error) {
	if len(it.extra) > 0 {
		ev := it.extra[0]
		it.extra = it.extra[1:]

		if len(it.extra) == 0 {
			it.extra = it.extraArr[:0]
		}
		// Two events aren't allowed to fall on the same timestamp in the new API,
		// but this may happen when we produce EvGoStatus events
		if ev.base.time <= it.lastTs {
			ev.base.time = it.lastTs + 1
		}
		it.lastTs = ev.base.time
		return ev, nil
	}

	oev, ok := it.events.Pop()
	if !ok {
		return Event{}, io.EOF
	}

	ev, err := it.convertEvent(oev)

	if err == errSkip {
		return it.next()
	} else if err != nil {
		return Event{}, err
	}

	// Two events aren't allowed to fall on the same timestamp in the new API,
	// but this may happen when we produce EvGoStatus events
	if ev.base.time <= it.lastTs {
		ev.base.time = it.lastTs + 1
	}
	it.lastTs = ev.base.time
	return ev, nil
}

var errSkip = errors.New("skip event")

// convertEvent converts an event from the old trace format to zero or more
// events in the new format. Most events translate 1 to 1. Some events don't
// result in an event right away, in which case convertEvent returns errSkip.
// Some events result in more than one new event; in this case, convertEvent
// returns the first event and stores additional events in it.extra. When
// encountering events that oldtrace shouldn't be able to emit, ocnvertEvent
// returns a descriptive error.
func (it *oldTraceConverter) convertEvent(ev *oldtrace.Event) (OUT Event, ERR error) {
	var mappedType event.Type
	var mappedArgs timedEventArgs
	copy(mappedArgs[:], ev.Args[:])

	switch ev.Type {
	case oldtrace.EvGomaxprocs:
		mappedType = go122.EvProcsChange
		if it.preInit {
			// The first EvGomaxprocs signals the end of trace initialization. At this point we've seen
			// all goroutines that already existed at trace begin.
			it.preInit = false
			for gid := range it.createdPreInit {
				// These are goroutines that already existed when tracing started but for which we
				// received neither GoWaiting, GoInSyscall, or GoStart. These are goroutines that are in
				// the states _Gidle or _Grunnable.
				it.extra = append(it.extra, Event{
					ctx: schedCtx{
						// G: GoID(gid),
						G: NoGoroutine,
						P: NoProc,
						M: NoThread,
					},
					table: it.evt,
					base: baseEvent{
						typ:  go122.EvGoStatus,
						time: Time(ev.Ts),
						args: timedEventArgs{uint64(gid), ^uint64(0), uint64(go122.GoRunnable)},
					},
				})
			}
			it.createdPreInit = nil
			return Event{}, errSkip
		}
	case oldtrace.EvProcStart:
		it.procMs[ProcID(ev.P)] = ThreadID(ev.Args[0])
		if _, ok := it.seenProcs[ProcID(ev.P)]; ok {
			mappedType = go122.EvProcStart
			mappedArgs = timedEventArgs{uint64(ev.P)}
		} else {
			it.seenProcs[ProcID(ev.P)] = struct{}{}
			mappedType = go122.EvProcStatus
			mappedArgs = timedEventArgs{uint64(ev.P), uint64(go122.ProcRunning)}
		}
	case oldtrace.EvProcStop:
		if _, ok := it.seenProcs[ProcID(ev.P)]; ok {
			mappedType = go122.EvProcStop
			mappedArgs = timedEventArgs{uint64(ev.P)}
		} else {
			it.seenProcs[ProcID(ev.P)] = struct{}{}
			mappedType = go122.EvProcStatus
			mappedArgs = timedEventArgs{uint64(ev.P), uint64(go122.ProcIdle)}
		}
	case oldtrace.EvGCStart:
		mappedType = go122.EvGCBegin
	case oldtrace.EvGCDone:
		mappedType = go122.EvGCEnd
	case oldtrace.EvSTWStart:
		sid := it.builtinToStringID[sSTWUnknown+it.trace.STWReason(ev.Args[0])]
		it.lastStwReason = sid
		mappedType = go122.EvSTWBegin
		mappedArgs = timedEventArgs{uint64(sid)}
	case oldtrace.EvSTWDone:
		mappedType = go122.EvSTWEnd
		mappedArgs = timedEventArgs{it.lastStwReason}
	case oldtrace.EvGCSweepStart:
		mappedType = go122.EvGCSweepBegin
	case oldtrace.EvGCSweepDone:
		mappedType = go122.EvGCSweepEnd
	case oldtrace.EvGoCreate:
		if it.preInit {
			it.createdPreInit[GoID(ev.Args[0])] = struct{}{}
			return Event{}, errSkip
		}
		mappedType = go122.EvGoCreate
	case oldtrace.EvGoStart:
		if it.preInit {
			mappedType = go122.EvGoStatus
			mappedArgs = timedEventArgs{ev.Args[0], ^uint64(0), uint64(go122.GoRunning)}
			delete(it.createdPreInit, GoID(ev.Args[0]))
		} else {
			mappedType = go122.EvGoStart
		}
	case oldtrace.EvGoStartLabel:
		it.extra = []Event{{
			ctx: schedCtx{
				G: GoID(ev.G),
				P: ProcID(ev.P),
				M: it.procMs[ProcID(ev.P)],
			},
			table: it.evt,
			base: baseEvent{
				typ:  go122.EvGoLabel,
				time: Time(ev.Ts),
				args: timedEventArgs{ev.Args[2]},
			},
		}}
		return Event{
			ctx: schedCtx{
				G: GoID(ev.G),
				P: ProcID(ev.P),
				M: it.procMs[ProcID(ev.P)],
			},
			table: it.evt,
			base: baseEvent{
				typ:  go122.EvGoStart,
				time: Time(ev.Ts),
				args: mappedArgs,
			},
		}, nil
	case oldtrace.EvGoEnd:
		mappedType = go122.EvGoDestroy
	case oldtrace.EvGoStop:
		mappedType = go122.EvGoBlock
		mappedArgs = timedEventArgs{uint64(it.builtinToStringID[sForever]), uint64(ev.StkID)}
	case oldtrace.EvGoSched:
		mappedType = go122.EvGoStop
		mappedArgs = timedEventArgs{uint64(it.builtinToStringID[sGosched]), uint64(ev.StkID)}
	case oldtrace.EvGoPreempt:
		mappedType = go122.EvGoStop
		mappedArgs = timedEventArgs{uint64(it.builtinToStringID[sPreempted]), uint64(ev.StkID)}
	case oldtrace.EvGoSleep:
		mappedType = go122.EvGoBlock
		mappedArgs = timedEventArgs{uint64(it.builtinToStringID[sSleep]), uint64(ev.StkID)}
	case oldtrace.EvGoBlock:
		mappedType = go122.EvGoBlock
		mappedArgs = timedEventArgs{uint64(it.builtinToStringID[sEmpty]), uint64(ev.StkID)}
	case oldtrace.EvGoUnblock:
		mappedType = go122.EvGoUnblock
	case oldtrace.EvGoBlockSend:
		mappedType = go122.EvGoBlock
		mappedArgs = timedEventArgs{uint64(it.builtinToStringID[sChanSend]), uint64(ev.StkID)}
	case oldtrace.EvGoBlockRecv:
		mappedType = go122.EvGoBlock
		mappedArgs = timedEventArgs{uint64(it.builtinToStringID[sChanRecv]), uint64(ev.StkID)}
	case oldtrace.EvGoBlockSelect:
		mappedType = go122.EvGoBlock
		mappedArgs = timedEventArgs{uint64(it.builtinToStringID[sSelect]), uint64(ev.StkID)}
	case oldtrace.EvGoBlockSync:
		mappedType = go122.EvGoBlock
		mappedArgs = timedEventArgs{uint64(it.builtinToStringID[sSync]), uint64(ev.StkID)}
	case oldtrace.EvGoBlockCond:
		mappedType = go122.EvGoBlock
		mappedArgs = timedEventArgs{uint64(it.builtinToStringID[sSyncCond]), uint64(ev.StkID)}
	case oldtrace.EvGoBlockNet:
		mappedType = go122.EvGoBlock
		mappedArgs = timedEventArgs{uint64(it.builtinToStringID[sNetwork]), uint64(ev.StkID)}
	case oldtrace.EvGoBlockGC:
		mappedType = go122.EvGoBlock
		mappedArgs = timedEventArgs{uint64(it.builtinToStringID[sMarkAssistWait]), uint64(ev.StkID)}
	case oldtrace.EvGoSysCall:
		// Look for the next event for the same G to determine if the syscall
		// blocked.
		blocked := false
		it.events.All()(func(nev *oldtrace.Event) bool {
			if nev.G != ev.G {
				return true
			}
			// After an EvGoSysCall, the next event on the same G will either be
			// EvGoSysBlock to denote a blocking syscall, or some other event
			// (or the end of the trace) if the syscall didn't block.
			if nev.Type == oldtrace.EvGoSysBlock {
				blocked = true
			}
			return false
		})
		if blocked {
			mappedType = go122.EvGoSyscallBegin
			mappedArgs = timedEventArgs{1: uint64(ev.StkID)}
		} else {
			// Convert the old instantaneous syscall event to a pair of syscall
			// begin and syscall end and give it the shortest possible duration,
			// 1ns.
			out1 := Event{
				ctx: schedCtx{
					G: GoID(ev.G),
					P: ProcID(ev.P),
					M: it.procMs[ProcID(ev.P)],
				},
				table: it.evt,
				base: baseEvent{
					typ:  go122.EvGoSyscallBegin,
					time: Time(ev.Ts),
					args: timedEventArgs{1: uint64(ev.StkID)},
				},
			}

			out2 := Event{
				ctx:   out1.ctx,
				table: it.evt,
				base: baseEvent{
					typ:  go122.EvGoSyscallEnd,
					time: Time(ev.Ts + 1),
					args: timedEventArgs{},
				},
			}

			it.extra = append(it.extra, out2)
			return out1, nil
		}

	case oldtrace.EvGoSysExit:
		mappedType = go122.EvGoSyscallEndBlocked
	case oldtrace.EvGoSysBlock:
		return Event{}, errSkip
	case oldtrace.EvGoWaiting:
		mappedType = go122.EvGoStatus
		mappedArgs = timedEventArgs{ev.Args[0], ^uint64(0), uint64(go122.GoWaiting)}
		delete(it.createdPreInit, GoID(ev.Args[0]))
	case oldtrace.EvGoInSyscall:
		mappedType = go122.EvGoStatus
		// In the new tracer, GoStatus with GoSyscall knows what thread the
		// syscall is on. In the old tracer, EvGoInSyscall doesn't contain that
		// information and all we can do here is specify NoThread.
		mappedArgs = timedEventArgs{ev.Args[0], ^uint64(0), uint64(go122.GoSyscall)}
		delete(it.createdPreInit, GoID(ev.Args[0]))
	case oldtrace.EvHeapAlloc:
		mappedType = go122.EvHeapAlloc
	case oldtrace.EvHeapGoal:
		mappedType = go122.EvHeapGoal
	case oldtrace.EvGCMarkAssistStart:
		mappedType = go122.EvGCMarkAssistBegin
	case oldtrace.EvGCMarkAssistDone:
		mappedType = go122.EvGCMarkAssistEnd
	case oldtrace.EvUserTaskCreate:
		mappedType = go122.EvUserTaskBegin
		parent := ev.Args[1]
		if parent == 0 {
			parent = uint64(NoTask)
		}
		mappedArgs = timedEventArgs{ev.Args[0], parent, ev.Args[2], uint64(ev.StkID)}
		name, _ := it.evt.strings.get(stringID(ev.Args[2]))
		it.tasks[TaskID(ev.Args[0])] = taskState{name: name, parentID: TaskID(ev.Args[1])}
	case oldtrace.EvUserTaskEnd:
		mappedType = go122.EvUserTaskEnd
		// Event.Task expects the parent and name to be smuggled in extra args
		// and as extra strings.
		ts, ok := it.tasks[TaskID(ev.Args[0])]
		if ok {
			delete(it.tasks, TaskID(ev.Args[0]))
			mappedArgs = timedEventArgs{
				ev.Args[0],
				ev.Args[1],
				uint64(ts.parentID),
				uint64(it.evt.addExtraString(ts.name)),
			}
		} else {
			mappedArgs = timedEventArgs{ev.Args[0], ev.Args[1], uint64(NoTask), uint64(it.evt.addExtraString(""))}
		}
	case oldtrace.EvUserRegion:
		switch ev.Args[1] {
		case 0: // start
			mappedType = go122.EvUserRegionBegin
		case 1: // end
			mappedType = go122.EvUserRegionEnd
		}
		mappedArgs = timedEventArgs{ev.Args[0], ev.Args[2], uint64(ev.StkID)}
	case oldtrace.EvUserLog:
		mappedType = go122.EvUserLog
		mappedArgs = timedEventArgs{ev.Args[0], ev.Args[1], it.inlineToStringID[ev.Args[3]], uint64(ev.StkID)}
	case oldtrace.EvCPUSample:
		mappedType = go122.EvCPUSample
		// When emitted by the Go 1.22 tracer, CPU samples have 5 arguments:
		// timestamp, M, P, G, stack. However, after they get turned into Event,
		// they have the arguments stack, M, P, G.
		//
		// In Go 1.21, CPU samples did not have Ms.
		mappedArgs = timedEventArgs{uint64(ev.StkID), ^uint64(0), uint64(ev.P), ev.G}
	default:
		return Event{}, fmt.Errorf("unexpected event type %v", ev.Type)
	}

	if oldtrace.EventDescriptions[ev.Type].Stack {
		if stackIDs := go122.Specs()[mappedType].StackIDs; len(stackIDs) > 0 {
			mappedArgs[stackIDs[0]-1] = uint64(ev.StkID)
		}
	}

	m := NoThread
	if ev.P != -1 && ev.Type != oldtrace.EvCPUSample {
		if t, ok := it.procMs[ProcID(ev.P)]; ok {
			m = ThreadID(t)
		}
	}
	if ev.Type == oldtrace.EvProcStop {
		delete(it.procMs, ProcID(ev.P))
	}
	g := GoID(ev.G)
	if g == 0 {
		g = NoGoroutine
	}
	out := Event{
		ctx: schedCtx{
			G: GoID(g),
			P: ProcID(ev.P),
			M: m,
		},
		table: it.evt,
		base: baseEvent{
			typ:  mappedType,
			time: Time(ev.Ts),
			args: mappedArgs,
		},
	}
	return out, nil
}

// convertOldFormat takes a fully loaded trace in the old trace format and
// returns an iterator over events in the new format.
func convertOldFormat(pr oldtrace.Trace) *oldTraceConverter {
	it := &oldTraceConverter{}
	it.init(pr)
	return it
}
