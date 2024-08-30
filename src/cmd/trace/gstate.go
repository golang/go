// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"internal/trace"
	"internal/trace/traceviewer"
	"internal/trace/traceviewer/format"
	"strings"
)

// resource is a generic constraint interface for resource IDs.
type resource interface {
	trace.GoID | trace.ProcID | trace.ThreadID
}

// noResource indicates the lack of a resource.
const noResource = -1

// gState represents the trace viewer state of a goroutine in a trace.
//
// The type parameter on this type is the resource which is used to construct
// a timeline of events. e.g. R=ProcID for a proc-oriented view, R=GoID for
// a goroutine-oriented view, etc.
type gState[R resource] struct {
	baseName  string
	named     bool   // Whether baseName has been set.
	label     string // EventLabel extension.
	isSystemG bool

	executing R // The resource this goroutine is executing on. (Could be itself.)

	// lastStopStack is the stack trace at the point of the last
	// call to the stop method. This tends to be a more reliable way
	// of picking up stack traces, since the parser doesn't provide
	// a stack for every state transition event.
	lastStopStack trace.Stack

	// activeRanges is the set of all active ranges on the goroutine.
	activeRanges map[string]activeRange

	// completedRanges is a list of ranges that completed since before the
	// goroutine stopped executing. These are flushed on every stop or block.
	completedRanges []completedRange

	// startRunningTime is the most recent event that caused a goroutine to
	// transition to GoRunning.
	startRunningTime trace.Time

	// startSyscall is the most recent event that caused a goroutine to
	// transition to GoSyscall.
	syscall struct {
		time   trace.Time
		stack  trace.Stack
		active bool
	}

	// startBlockReason is the StateTransition.Reason of the most recent
	// event that caused a goroutine to transition to GoWaiting.
	startBlockReason string

	// startCause is the event that allowed this goroutine to start running.
	// It's used to generate flow events. This is typically something like
	// an unblock event or a goroutine creation event.
	//
	// startCause.resource is the resource on which startCause happened, but is
	// listed separately because the cause may have happened on a resource that
	// isn't R (or perhaps on some abstract nebulous resource, like trace.NetpollP).
	startCause struct {
		time     trace.Time
		name     string
		resource uint64
		stack    trace.Stack
	}
}

// newGState constructs a new goroutine state for the goroutine
// identified by the provided ID.
func newGState[R resource](goID trace.GoID) *gState[R] {
	return &gState[R]{
		baseName:     fmt.Sprintf("G%d", goID),
		executing:    R(noResource),
		activeRanges: make(map[string]activeRange),
	}
}

// augmentName attempts to use stk to augment the name of the goroutine
// with stack information. This stack must be related to the goroutine
// in some way, but it doesn't really matter which stack.
func (gs *gState[R]) augmentName(stk trace.Stack) {
	if gs.named {
		return
	}
	if stk == trace.NoStack {
		return
	}
	name := lastFunc(stk)
	gs.baseName += fmt.Sprintf(" %s", name)
	gs.named = true
	gs.isSystemG = trace.IsSystemGoroutine(name)
}

// setLabel adds an additional label to the goroutine's name.
func (gs *gState[R]) setLabel(label string) {
	gs.label = label
}

// name returns a name for the goroutine.
func (gs *gState[R]) name() string {
	name := gs.baseName
	if gs.label != "" {
		name += " (" + gs.label + ")"
	}
	return name
}

// setStartCause sets the reason a goroutine will be allowed to start soon.
// For example, via unblocking or exiting a blocked syscall.
func (gs *gState[R]) setStartCause(ts trace.Time, name string, resource uint64, stack trace.Stack) {
	gs.startCause.time = ts
	gs.startCause.name = name
	gs.startCause.resource = resource
	gs.startCause.stack = stack
}

// created indicates that this goroutine was just created by the provided creator.
func (gs *gState[R]) created(ts trace.Time, creator R, stack trace.Stack) {
	if creator == R(noResource) {
		return
	}
	gs.setStartCause(ts, "go", uint64(creator), stack)
}

// start indicates that a goroutine has started running on a proc.
func (gs *gState[R]) start(ts trace.Time, resource R, ctx *traceContext) {
	// Set the time for all the active ranges.
	for name := range gs.activeRanges {
		gs.activeRanges[name] = activeRange{ts, trace.NoStack}
	}

	if gs.startCause.name != "" {
		// It has a start cause. Emit a flow event.
		ctx.Arrow(traceviewer.ArrowEvent{
			Name:         gs.startCause.name,
			Start:        ctx.elapsed(gs.startCause.time),
			End:          ctx.elapsed(ts),
			FromResource: uint64(gs.startCause.resource),
			ToResource:   uint64(resource),
			FromStack:    ctx.Stack(viewerFrames(gs.startCause.stack)),
		})
		gs.startCause.time = 0
		gs.startCause.name = ""
		gs.startCause.resource = 0
		gs.startCause.stack = trace.NoStack
	}
	gs.executing = resource
	gs.startRunningTime = ts
}

// syscallBegin indicates that the goroutine entered a syscall on a proc.
func (gs *gState[R]) syscallBegin(ts trace.Time, resource R, stack trace.Stack) {
	gs.syscall.time = ts
	gs.syscall.stack = stack
	gs.syscall.active = true
	if gs.executing == R(noResource) {
		gs.executing = resource
		gs.startRunningTime = ts
	}
}

// syscallEnd ends the syscall slice, wherever the syscall is at. This is orthogonal
// to blockedSyscallEnd -- both must be called when a syscall ends and that syscall
// blocked. They're kept separate because syscallEnd indicates the point at which the
// goroutine is no longer executing on the resource (e.g. a proc) whereas blockedSyscallEnd
// is the point at which the goroutine actually exited the syscall regardless of which
// resource that happened on.
func (gs *gState[R]) syscallEnd(ts trace.Time, blocked bool, ctx *traceContext) {
	if !gs.syscall.active {
		return
	}
	blockString := "no"
	if blocked {
		blockString = "yes"
	}
	gs.completedRanges = append(gs.completedRanges, completedRange{
		name:       "syscall",
		startTime:  gs.syscall.time,
		endTime:    ts,
		startStack: gs.syscall.stack,
		arg:        format.BlockedArg{Blocked: blockString},
	})
	gs.syscall.active = false
	gs.syscall.time = 0
	gs.syscall.stack = trace.NoStack
}

// blockedSyscallEnd indicates the point at which the blocked syscall ended. This is distinct
// and orthogonal to syscallEnd; both must be called if the syscall blocked. This sets up an instant
// to emit a flow event from, indicating explicitly that this goroutine was unblocked by the system.
func (gs *gState[R]) blockedSyscallEnd(ts trace.Time, stack trace.Stack, ctx *traceContext) {
	name := "exit blocked syscall"
	gs.setStartCause(ts, name, trace.SyscallP, stack)

	// Emit an syscall exit instant event for the "Syscall" lane.
	ctx.Instant(traceviewer.InstantEvent{
		Name:     name,
		Ts:       ctx.elapsed(ts),
		Resource: trace.SyscallP,
		Stack:    ctx.Stack(viewerFrames(stack)),
	})
}

// unblock indicates that the goroutine gs represents has been unblocked.
func (gs *gState[R]) unblock(ts trace.Time, stack trace.Stack, resource R, ctx *traceContext) {
	name := "unblock"
	viewerResource := uint64(resource)
	if gs.startBlockReason != "" {
		name = fmt.Sprintf("%s (%s)", name, gs.startBlockReason)
	}
	if strings.Contains(gs.startBlockReason, "network") {
		// Attribute the network instant to the nebulous "NetpollP" if
		// resource isn't a thread, because there's a good chance that
		// resource isn't going to be valid in this case.
		//
		// TODO(mknyszek): Handle this invalidness in a more general way.
		if _, ok := any(resource).(trace.ThreadID); !ok {
			// Emit an unblock instant event for the "Network" lane.
			viewerResource = trace.NetpollP
		}
		ctx.Instant(traceviewer.InstantEvent{
			Name:     name,
			Ts:       ctx.elapsed(ts),
			Resource: viewerResource,
			Stack:    ctx.Stack(viewerFrames(stack)),
		})
	}
	gs.startBlockReason = ""
	if viewerResource != 0 {
		gs.setStartCause(ts, name, viewerResource, stack)
	}
}

// block indicates that the goroutine has stopped executing on a proc -- specifically,
// it blocked for some reason.
func (gs *gState[R]) block(ts trace.Time, stack trace.Stack, reason string, ctx *traceContext) {
	gs.startBlockReason = reason
	gs.stop(ts, stack, ctx)
}

// stop indicates that the goroutine has stopped executing on a proc.
func (gs *gState[R]) stop(ts trace.Time, stack trace.Stack, ctx *traceContext) {
	// Emit the execution time slice.
	var stk int
	if gs.lastStopStack != trace.NoStack {
		stk = ctx.Stack(viewerFrames(gs.lastStopStack))
	}
	// Check invariants.
	if gs.startRunningTime == 0 {
		panic("silently broken trace or generator invariant (startRunningTime != 0) not held")
	}
	if gs.executing == R(noResource) {
		panic("non-executing goroutine stopped")
	}
	ctx.Slice(traceviewer.SliceEvent{
		Name:     gs.name(),
		Ts:       ctx.elapsed(gs.startRunningTime),
		Dur:      ts.Sub(gs.startRunningTime),
		Resource: uint64(gs.executing),
		Stack:    stk,
	})

	// Flush completed ranges.
	for _, cr := range gs.completedRanges {
		ctx.Slice(traceviewer.SliceEvent{
			Name:     cr.name,
			Ts:       ctx.elapsed(cr.startTime),
			Dur:      cr.endTime.Sub(cr.startTime),
			Resource: uint64(gs.executing),
			Stack:    ctx.Stack(viewerFrames(cr.startStack)),
			EndStack: ctx.Stack(viewerFrames(cr.endStack)),
			Arg:      cr.arg,
		})
	}
	gs.completedRanges = gs.completedRanges[:0]

	// Continue in-progress ranges.
	for name, r := range gs.activeRanges {
		// Check invariant.
		if r.time == 0 {
			panic("silently broken trace or generator invariant (activeRanges time != 0) not held")
		}
		ctx.Slice(traceviewer.SliceEvent{
			Name:     name,
			Ts:       ctx.elapsed(r.time),
			Dur:      ts.Sub(r.time),
			Resource: uint64(gs.executing),
			Stack:    ctx.Stack(viewerFrames(r.stack)),
		})
	}

	// Clear the range info.
	for name := range gs.activeRanges {
		gs.activeRanges[name] = activeRange{0, trace.NoStack}
	}

	gs.startRunningTime = 0
	gs.lastStopStack = stack
	gs.executing = R(noResource)
}

// finalize writes out any in-progress slices as if the goroutine stopped.
// This must only be used once the trace has been fully processed and no
// further events will be processed. This method may leave the gState in
// an inconsistent state.
func (gs *gState[R]) finish(ctx *traceContext) {
	if gs.executing != R(noResource) {
		gs.syscallEnd(ctx.endTime, false, ctx)
		gs.stop(ctx.endTime, trace.NoStack, ctx)
	}
}

// rangeBegin indicates the start of a special range of time.
func (gs *gState[R]) rangeBegin(ts trace.Time, name string, stack trace.Stack) {
	if gs.executing != R(noResource) {
		// If we're executing, start the slice from here.
		gs.activeRanges[name] = activeRange{ts, stack}
	} else {
		// If the goroutine isn't executing, there's no place for
		// us to create a slice from. Wait until it starts executing.
		gs.activeRanges[name] = activeRange{0, stack}
	}
}

// rangeActive indicates that a special range of time has been in progress.
func (gs *gState[R]) rangeActive(name string) {
	if gs.executing != R(noResource) {
		// If we're executing, and the range is active, then start
		// from wherever the goroutine started running from.
		gs.activeRanges[name] = activeRange{gs.startRunningTime, trace.NoStack}
	} else {
		// If the goroutine isn't executing, there's no place for
		// us to create a slice from. Wait until it starts executing.
		gs.activeRanges[name] = activeRange{0, trace.NoStack}
	}
}

// rangeEnd indicates the end of a special range of time.
func (gs *gState[R]) rangeEnd(ts trace.Time, name string, stack trace.Stack, ctx *traceContext) {
	if gs.executing != R(noResource) {
		r := gs.activeRanges[name]
		gs.completedRanges = append(gs.completedRanges, completedRange{
			name:       name,
			startTime:  r.time,
			endTime:    ts,
			startStack: r.stack,
			endStack:   stack,
		})
	}
	delete(gs.activeRanges, name)
}

func lastFunc(s trace.Stack) string {
	var last trace.StackFrame
	s.Frames(func(f trace.StackFrame) bool {
		last = f
		return true
	})
	return last.Func
}
