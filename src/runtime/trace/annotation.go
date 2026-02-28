// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"context"
	"fmt"
	"sync/atomic"
	_ "unsafe"
)

type traceContextKey struct{}

// NewTask creates a task instance with the type taskType and returns
// it along with a Context that carries the task.
// If the input context contains a task, the new task is its subtask.
//
// The taskType is used to classify task instances. Analysis tools
// like the Go execution tracer may assume there are only a bounded
// number of unique task types in the system.
//
// The returned end function is used to mark the task's end.
// The trace tool measures task latency as the time between task creation
// and when the end function is called, and provides the latency
// distribution per task type.
// If the end function is called multiple times, only the first
// call is used in the latency measurement.
//
//	ctx, task := trace.NewTask(ctx, "awesomeTask")
//	trace.WithRegion(ctx, "preparation", prepWork)
//	// preparation of the task
//	go func() {  // continue processing the task in a separate goroutine.
//	    defer task.End()
//	    trace.WithRegion(ctx, "remainingWork", remainingWork)
//	}()
func NewTask(pctx context.Context, taskType string) (ctx context.Context, task *Task) {
	pid := fromContext(pctx).id
	id := newID()
	userTaskCreate(id, pid, taskType)
	s := &Task{id: id}
	return context.WithValue(pctx, traceContextKey{}, s), s

	// We allocate a new task and the end function even when
	// the tracing is disabled because the context and the detach
	// function can be used across trace enable/disable boundaries,
	// which complicates the problem.
	//
	// For example, consider the following scenario:
	//   - trace is enabled.
	//   - trace.WithRegion is called, so a new context ctx
	//     with a new region is created.
	//   - trace is disabled.
	//   - trace is enabled again.
	//   - trace APIs with the ctx is called. Is the ID in the task
	//   a valid one to use?
	//
	// TODO(hyangah): reduce the overhead at least when
	// tracing is disabled. Maybe the id can embed a tracing
	// round number and ignore ids generated from previous
	// tracing round.
}

func fromContext(ctx context.Context) *Task {
	if s, ok := ctx.Value(traceContextKey{}).(*Task); ok {
		return s
	}
	return &bgTask
}

// Task is a data type for tracing a user-defined, logical operation.
type Task struct {
	id uint64
	// TODO(hyangah): record parent id?
}

// End marks the end of the operation represented by the Task.
func (t *Task) End() {
	userTaskEnd(t.id)
}

var lastTaskID uint64 = 0 // task id issued last time

func newID() uint64 {
	// TODO(hyangah): use per-P cache
	return atomic.AddUint64(&lastTaskID, 1)
}

var bgTask = Task{id: uint64(0)}

// Log emits a one-off event with the given category and message.
// Category can be empty and the API assumes there are only a handful of
// unique categories in the system.
func Log(ctx context.Context, category, message string) {
	id := fromContext(ctx).id
	userLog(id, category, message)
}

// Logf is like Log, but the value is formatted using the specified format spec.
func Logf(ctx context.Context, category, format string, args ...any) {
	if IsEnabled() {
		// Ideally this should be just Log, but that will
		// add one more frame in the stack trace.
		id := fromContext(ctx).id
		userLog(id, category, fmt.Sprintf(format, args...))
	}
}

const (
	regionStartCode = uint64(0)
	regionEndCode   = uint64(1)
)

// WithRegion starts a region associated with its calling goroutine, runs fn,
// and then ends the region. If the context carries a task, the region is
// associated with the task. Otherwise, the region is attached to the background
// task.
//
// The regionType is used to classify regions, so there should be only a
// handful of unique region types.
func WithRegion(ctx context.Context, regionType string, fn func()) {
	// NOTE:
	// WithRegion helps avoiding misuse of the API but in practice,
	// this is very restrictive:
	// - Use of WithRegion makes the stack traces captured from
	//   region start and end are identical.
	// - Refactoring the existing code to use WithRegion is sometimes
	//   hard and makes the code less readable.
	//     e.g. code block nested deep in the loop with various
	//          exit point with return values
	// - Refactoring the code to use this API with closure can
	//   cause different GC behavior such as retaining some parameters
	//   longer.
	// This causes more churns in code than I hoped, and sometimes
	// makes the code less readable.

	id := fromContext(ctx).id
	userRegion(id, regionStartCode, regionType)
	defer userRegion(id, regionEndCode, regionType)
	fn()
}

// StartRegion starts a region and returns a function for marking the
// end of the region. The returned Region's End function must be called
// from the same goroutine where the region was started.
// Within each goroutine, regions must nest. That is, regions started
// after this region must be ended before this region can be ended.
// Recommended usage is
//
//	defer trace.StartRegion(ctx, "myTracedRegion").End()
func StartRegion(ctx context.Context, regionType string) *Region {
	if !IsEnabled() {
		return noopRegion
	}
	id := fromContext(ctx).id
	userRegion(id, regionStartCode, regionType)
	return &Region{id, regionType}
}

// Region is a region of code whose execution time interval is traced.
type Region struct {
	id         uint64
	regionType string
}

var noopRegion = &Region{}

// End marks the end of the traced code region.
func (r *Region) End() {
	if r == noopRegion {
		return
	}
	userRegion(r.id, regionEndCode, r.regionType)
}

// IsEnabled reports whether tracing is enabled.
// The information is advisory only. The tracing status
// may have changed by the time this function returns.
func IsEnabled() bool {
	enabled := atomic.LoadInt32(&tracing.enabled)
	return enabled == 1
}

//
// Function bodies are defined in runtime/trace.go
//

// emits UserTaskCreate event.
func userTaskCreate(id, parentID uint64, taskType string)

// emits UserTaskEnd event.
func userTaskEnd(id uint64)

// emits UserRegion event.
func userRegion(id, mode uint64, regionType string)

// emits UserLog event.
func userLog(id uint64, category, message string)
