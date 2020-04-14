// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package export

import (
	"context"
	"fmt"
	"io"
	"sync"

	"golang.org/x/tools/internal/telemetry/event"
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
	buffer     [128]byte
	writer     io.Writer
	onlyErrors bool
}

func (w *logWriter) ProcessEvent(ctx context.Context, ev event.Event, tagMap event.TagMap) context.Context {
	switch {
	case ev.IsLog():
		if w.onlyErrors && event.Err.Get(tagMap) == nil {
			return ctx
		}
		w.mu.Lock()
		defer w.mu.Unlock()

		buf := w.buffer[:0]
		if !ev.At.IsZero() {
			w.writer.Write(ev.At.AppendFormat(buf, "2006/01/02 15:04:05 "))
		}
		msg := event.Msg.Get(tagMap)
		io.WriteString(w.writer, msg)
		if err := event.Err.Get(tagMap); err != nil {
			io.WriteString(w.writer, ": ")
			io.WriteString(w.writer, err.Error())
		}
		for index := 0; ev.Valid(index); index++ {
			tag := ev.Tag(index)
			if !tag.Valid() || tag.Key() == event.Msg || tag.Key() == event.Err {
				continue
			}
			io.WriteString(w.writer, "\n\t")
			io.WriteString(w.writer, tag.Key().Name())
			io.WriteString(w.writer, "=")
			tag.Key().Format(w.writer, buf, tag)
		}
		io.WriteString(w.writer, "\n")

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
