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

func (w *logWriter) StartSpan(context.Context, *telemetry.Span)  {}
func (w *logWriter) FinishSpan(context.Context, *telemetry.Span) {}
func (w *logWriter) Log(ctx context.Context, event telemetry.Event) {
	if event.Error == nil {
		// we only log errors by default
		return
	}
	fmt.Fprintf(w.writer, "%v\n", event)
}
func (w *logWriter) Metric(context.Context, telemetry.MetricData) {}
func (w *logWriter) Flush()                                       {}
