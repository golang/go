// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package export

import (
	"context"
	"fmt"
	"io"
	"os"

	"golang.org/x/tools/internal/telemetry/event"
)

func init() {
	event.SetExporter(LogWriter(os.Stderr, true))
}

// LogWriter returns an Exporter that logs events to the supplied writer.
// If onlyErrors is true it does not log any event that did not have an
// associated error.
// It ignores all telemetry other than log events.
func LogWriter(w io.Writer, onlyErrors bool) event.Exporter {
	return &logWriter{writer: w, onlyErrors: onlyErrors}
}

type logWriter struct {
	writer     io.Writer
	onlyErrors bool
}

func (w *logWriter) ProcessEvent(ctx context.Context, ev event.Event) context.Context {
	switch {
	case ev.IsLog():
		if w.onlyErrors && ev.Error == nil {
			return ctx
		}
		fmt.Fprintf(w.writer, "%v\n", ev)
	case ev.IsStartSpan():
		if span := GetSpan(ctx); span != nil {
			fmt.Fprintf(w.writer, "start: %v %v", span.Name, span.ID)
			if span.ParentID.IsValid() {
				fmt.Fprintf(w.writer, "[%v]", span.ParentID)
			}
		}
	case ev.IsEndSpan():
		if span := GetSpan(ctx); span != nil {
			fmt.Fprintf(w.writer, "finish: %v %v", span.Name, span.ID)
		}
	}
	return ctx
}

func (w *logWriter) Metric(context.Context, event.MetricData) {}
