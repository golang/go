// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package export

import (
	"context"
	"fmt"
	"io"
	"sync"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/core"
	"golang.org/x/tools/internal/event/label"
)

// LogWriter returns an Exporter that logs events to the supplied writer.
// If onlyErrors is true it does not log any event that did not have an
// associated error.
// It ignores all telemetry other than log events.
func LogWriter(w io.Writer, onlyErrors bool) event.Exporter {
	lw := &logWriter{writer: w, onlyErrors: onlyErrors}
	return lw.ProcessEvent
}

type logWriter struct {
	mu         sync.Mutex
	printer    Printer
	writer     io.Writer
	onlyErrors bool
}

func (w *logWriter) ProcessEvent(ctx context.Context, ev core.Event, lm label.Map) context.Context {
	switch {
	case event.IsLog(ev):
		if w.onlyErrors && !event.IsError(ev) {
			return ctx
		}
		w.mu.Lock()
		defer w.mu.Unlock()
		w.printer.WriteEvent(w.writer, ev, lm)

	case event.IsStart(ev):
		if span := GetSpan(ctx); span != nil {
			fmt.Fprintf(w.writer, "start: %v %v", span.Name, span.ID)
			if span.ParentID.IsValid() {
				fmt.Fprintf(w.writer, "[%v]", span.ParentID)
			}
		}
	case event.IsEnd(ev):
		if span := GetSpan(ctx); span != nil {
			fmt.Fprintf(w.writer, "finish: %v %v", span.Name, span.ID)
		}
	}
	return ctx
}
