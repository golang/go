// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"internal/trace"
	"internal/trace/traceviewer"
	tracev2 "internal/trace/v2"
	"time"
)

// viewerFrames returns the frames of the stack of ev. The given frame slice is
// used to store the frames to reduce allocations.
func viewerFrames(stk tracev2.Stack) []*trace.Frame {
	var frames []*trace.Frame
	stk.Frames(func(f tracev2.StackFrame) bool {
		frames = append(frames, &trace.Frame{
			PC:   f.PC,
			Fn:   f.Func,
			File: f.File,
			Line: int(f.Line),
		})
		return true
	})
	return frames
}

func viewerGState(state tracev2.GoState, inMarkAssist bool) traceviewer.GState {
	switch state {
	case tracev2.GoUndetermined:
		return traceviewer.GDead
	case tracev2.GoNotExist:
		return traceviewer.GDead
	case tracev2.GoRunnable:
		return traceviewer.GRunnable
	case tracev2.GoRunning:
		return traceviewer.GRunning
	case tracev2.GoWaiting:
		if inMarkAssist {
			return traceviewer.GWaitingGC
		}
		return traceviewer.GWaiting
	case tracev2.GoSyscall:
		// N.B. A goroutine in a syscall is considered "executing" (state.Executing() == true).
		return traceviewer.GRunning
	default:
		panic(fmt.Sprintf("unknown GoState: %s", state.String()))
	}
}

func viewerTime(t time.Duration) float64 {
	return float64(t) / float64(time.Microsecond)
}
