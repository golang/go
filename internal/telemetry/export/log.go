// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package export

import (
	"context"
	"fmt"
	"io"

	"golang.org/x/tools/internal/telemetry"
)

// LogWriter returns an observer that logs events to the supplied writer.
// If onlyErrors is true it does not log any event that did not have an
// associated error.
// It ignores all telemetry other than log events.
func LogWriter(w io.Writer, onlyErrors bool) Exporter {
	return &logWriter{writer: w, onlyErrors: onlyErrors}
}

type logWriter struct {
	writer     io.Writer
	onlyErrors bool
}

func (w *logWriter) ProcessEvent(ctx context.Context, event telemetry.Event) context.Context {
	switch event.Type {
	case telemetry.EventLog:
		if w.onlyErrors && event.Error == nil {
			return ctx
		}
		fmt.Fprintf(w.writer, "%v\n", event)
	case telemetry.EventStartSpan:
		if span := GetSpan(ctx); span != nil {
			fmt.Fprintf(w.writer, "start: %v %v", span.Name, span.ID)
			if span.ParentID.IsValid() {
				fmt.Fprintf(w.writer, "[%v]", span.ParentID)
			}
		}
	case telemetry.EventEndSpan:
		if span := GetSpan(ctx); span != nil {
			fmt.Fprintf(w.writer, "finish: %v %v", span.Name, span.ID)
		}
	}
	return ctx
}
func (w *logWriter) Metric(context.Context, telemetry.MetricData) {}
