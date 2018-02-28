package trace

import (
	"context"
	"fmt"
	"sync/atomic"
	_ "unsafe"
)

type traceContextKey struct{}

// NewContext creates a child context with a new task instance with
// the type taskType. If the input context contains a task, the
// new task is its subtask.
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
//   ctx, taskEnd := trace.NewContext(ctx, "awesome task")
//   trace.WithSpan(ctx, prepWork)
//   // preparation of the task
//   go func() {  // continue processing the task in a separate goroutine.
//       defer taskEnd()
//       trace.WithSpan(ctx, remainingWork)
//   }
func NewContext(pctx context.Context, taskType string) (ctx context.Context, end func()) {
	pid := fromContext(pctx).id
	id := newID()
	userTaskCreate(id, pid, taskType)
	s := &task{id: id}
	return context.WithValue(pctx, traceContextKey{}, s), func() {
		userTaskEnd(id)
	}

	// We allocate a new task and the end function even when
	// the tracing is disabled because the context and the detach
	// function can be used across trace enable/disable boundaries,
	// which complicates the problem.
	//
	// For example, consider the following scenario:
	//   - trace is enabled.
	//   - trace.WithSpan is called, so a new context ctx
	//     with a new span is created.
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

func fromContext(ctx context.Context) *task {
	if s, ok := ctx.Value(traceContextKey{}).(*task); ok {
		return s
	}
	return &bgTask
}

type task struct {
	id uint64
	// TODO(hyangah): record parent id?
}

var lastTaskID uint64 = 0 // task id issued last time

func newID() uint64 {
	// TODO(hyangah): use per-P cache
	return atomic.AddUint64(&lastTaskID, 1)
}

var bgTask = task{id: uint64(0)}

// Log emits a one-off event with the given category and message.
// Category can be empty and the API assumes there are only a handful of
// unique categories in the system.
func Log(ctx context.Context, category, message string) {
	id := fromContext(ctx).id
	userLog(id, category, message)
}

// Logf is like Log, but the value is formatted using the specified format spec.
func Logf(ctx context.Context, category, format string, args ...interface{}) {
	if IsEnabled() {
		// Ideally this should be just Log, but that will
		// add one more frame in the stack trace.
		id := fromContext(ctx).id
		userLog(id, category, fmt.Sprintf(format, args...))
	}
}

const (
	spanStartCode = uint64(0)
	spanEndCode   = uint64(1)
)

// WithSpan starts a span associated with its calling goroutine, runs fn,
// and then ends the span. If the context carries a task, the span is
// attached to the task. Otherwise, the span is attached to the background
// task.
//
// The spanType is used to classify spans, so there should be only a
// handful of unique span types.
func WithSpan(ctx context.Context, spanType string, fn func(context.Context)) {
	// NOTE:
	// WithSpan helps avoiding misuse of the API but in practice,
	// this is very restrictive:
	// - Use of WithSpan makes the stack traces captured from
	//   span start and end are identical.
	// - Refactoring the existing code to use WithSpan is sometimes
	//   hard and makes the code less readable.
	//     e.g. code block nested deep in the loop with various
	//          exit point with return values
	// - Refactoring the code to use this API with closure can
	//   cause different GC behavior such as retaining some parameters
	//   longer.
	// This causes more churns in code than I hoped, and sometimes
	// makes the code less readable.

	id := fromContext(ctx).id
	userSpan(id, spanStartCode, spanType)
	defer userSpan(id, spanEndCode, spanType)
	fn(ctx)
}

// StartSpan starts a span and returns a function for marking the
// end of the span. The span end function must be called from the
// same goroutine where the span was started.
// Within each goroutine, spans must nest. That is, spans started
// after this span must be ended before this span can be ended.
// Callers are encouraged to instead use WithSpan when possible,
// since it naturally satisfies these restrictions.
func StartSpan(ctx context.Context, spanType string) func() {
	id := fromContext(ctx).id
	userSpan(id, spanStartCode, spanType)
	return func() { userSpan(id, spanEndCode, spanType) }
}

// IsEnabled returns whether tracing is enabled.
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

// emits UserSpan event.
func userSpan(id, mode uint64, spanType string)

// emits UserLog event.
func userLog(id uint64, category, message string)
