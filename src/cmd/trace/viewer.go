// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"internal/trace"
	"internal/trace/traceviewer"
	"time"
)

// viewerFrames returns the frames of the stack of ev. The given frame slice is
// used to store the frames to reduce allocations.
func viewerFrames(stk trace.Stack) []*trace.Frame {
	var frames []*trace.Frame
	for f := range stk.Frames() {
		frames = append(frames, &trace.Frame{
			PC:   f.PC,
			Fn:   f.Func,
			File: f.File,
			Line: int(f.Line),
		})
	}
	return frames
}

func viewerGState(state trace.GoState, inMarkAssist bool) traceviewer.GState {
	switch state {
	case trace.GoUndetermined:
		return traceviewer.GDead
	case trace.GoNotExist:
		return traceviewer.GDead
	case trace.GoRunnable:
		return traceviewer.GRunnable
	case trace.GoRunning:
		return traceviewer.GRunning
	case trace.GoWaiting:
		if inMarkAssist {
			return traceviewer.GWaitingGC
		}
		return traceviewer.GWaiting
	case trace.GoSyscall:
		// N.B. A goroutine in a syscall is considered "executing" (state.Executing() == true).
		return traceviewer.GRunning
	default:
		panic(fmt.Sprintf("unknown GoState: %s", state.String()))
	}
}

func viewerTime(t time.Duration) float64 {
	return float64(t) / float64(time.Microsecond)
}
