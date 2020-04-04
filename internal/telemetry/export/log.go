// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package export

import (
	"context"
	"fmt"
	"io"
	"strconv"
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
			if !tag.Valid() || tag.Key == event.Msg || tag.Key == event.Err {
				continue
			}
			io.WriteString(w.writer, "\n\t")
			io.WriteString(w.writer, tag.Key.Name())
			io.WriteString(w.writer, "=")
			switch key := tag.Key.(type) {
			case *event.IntKey:
				w.writer.Write(strconv.AppendInt(buf, int64(key.From(tag)), 10))
			case *event.Int8Key:
				w.writer.Write(strconv.AppendInt(buf, int64(key.From(tag)), 10))
			case *event.Int16Key:
				w.writer.Write(strconv.AppendInt(buf, int64(key.From(tag)), 10))
			case *event.Int32Key:
				w.writer.Write(strconv.AppendInt(buf, int64(key.From(tag)), 10))
			case *event.Int64Key:
				w.writer.Write(strconv.AppendInt(buf, key.From(tag), 10))
			case *event.UIntKey:
				w.writer.Write(strconv.AppendUint(buf, uint64(key.From(tag)), 10))
			case *event.UInt8Key:
				w.writer.Write(strconv.AppendUint(buf, uint64(key.From(tag)), 10))
			case *event.UInt16Key:
				w.writer.Write(strconv.AppendUint(buf, uint64(key.From(tag)), 10))
			case *event.UInt32Key:
				w.writer.Write(strconv.AppendUint(buf, uint64(key.From(tag)), 10))
			case *event.UInt64Key:
				w.writer.Write(strconv.AppendUint(buf, key.From(tag), 10))
			case *event.Float32Key:
				w.writer.Write(strconv.AppendFloat(buf, float64(key.From(tag)), 'E', -1, 32))
			case *event.Float64Key:
				w.writer.Write(strconv.AppendFloat(buf, key.From(tag), 'E', -1, 64))
			case *event.BooleanKey:
				w.writer.Write(strconv.AppendBool(buf, key.From(tag)))
			case *event.StringKey:
				w.writer.Write(strconv.AppendQuote(buf, key.From(tag)))
			case *event.ErrorKey:
				io.WriteString(w.writer, key.From(tag).Error())
			case *event.ValueKey:
				fmt.Fprint(w.writer, key.From(tag))
			default:
				fmt.Fprintf(w.writer, `"invalid key type %T"`, key)
			}
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
