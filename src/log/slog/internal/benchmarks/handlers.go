// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package benchmarks

// Handlers for benchmarking.

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"log/slog/internal/buffer"
	"strconv"
	"time"
)

// A fastTextHandler writes a Record to an io.Writer in a format similar to
// slog.TextHandler, but without quoting or locking. It has a few other
// performance-motivated shortcuts, like writing times as seconds since the
// epoch instead of strings.
//
// It is intended to represent a high-performance Handler that synchronously
// writes text (as opposed to binary).
type fastTextHandler struct {
	w io.Writer
}

func newFastTextHandler(w io.Writer) slog.Handler {
	return &fastTextHandler{w: w}
}

func (h *fastTextHandler) Enabled(context.Context, slog.Level) bool { return true }

func (h *fastTextHandler) Handle(_ context.Context, r slog.Record) error {
	buf := buffer.New()
	defer buf.Free()

	if !r.Time.IsZero() {
		buf.WriteString("time=")
		h.appendTime(buf, r.Time)
		buf.WriteByte(' ')
	}
	buf.WriteString("level=")
	*buf = strconv.AppendInt(*buf, int64(r.Level), 10)
	buf.WriteByte(' ')
	buf.WriteString("msg=")
	buf.WriteString(r.Message)
	r.Attrs(func(a slog.Attr) bool {
		buf.WriteByte(' ')
		buf.WriteString(a.Key)
		buf.WriteByte('=')
		h.appendValue(buf, a.Value)
		return true
	})
	buf.WriteByte('\n')
	_, err := h.w.Write(*buf)
	return err
}

func (h *fastTextHandler) appendValue(buf *buffer.Buffer, v slog.Value) {
	switch v.Kind() {
	case slog.KindString:
		buf.WriteString(v.String())
	case slog.KindInt64:
		*buf = strconv.AppendInt(*buf, v.Int64(), 10)
	case slog.KindUint64:
		*buf = strconv.AppendUint(*buf, v.Uint64(), 10)
	case slog.KindFloat64:
		*buf = strconv.AppendFloat(*buf, v.Float64(), 'g', -1, 64)
	case slog.KindBool:
		*buf = strconv.AppendBool(*buf, v.Bool())
	case slog.KindDuration:
		*buf = strconv.AppendInt(*buf, v.Duration().Nanoseconds(), 10)
	case slog.KindTime:
		h.appendTime(buf, v.Time())
	case slog.KindAny:
		a := v.Any()
		switch a := a.(type) {
		case error:
			buf.WriteString(a.Error())
		default:
			fmt.Fprint(buf, a)
		}
	default:
		panic(fmt.Sprintf("bad kind: %s", v.Kind()))
	}
}

func (h *fastTextHandler) appendTime(buf *buffer.Buffer, t time.Time) {
	*buf = strconv.AppendInt(*buf, t.Unix(), 10)
}

func (h *fastTextHandler) WithAttrs([]slog.Attr) slog.Handler {
	panic("fastTextHandler: With unimplemented")
}

func (*fastTextHandler) WithGroup(string) slog.Handler {
	panic("fastTextHandler: WithGroup unimplemented")
}

// An asyncHandler simulates a Handler that passes Records to a
// background goroutine for processing.
// Because sending to a channel can be expensive due to locking,
// we simulate a lock-free queue by adding the Record to a ring buffer.
// Omitting the locking makes this little more than a copy of the Record,
// but that is a worthwhile thing to measure because Records are on the large
// side. Since nothing actually reads from the ring buffer, it can handle an
// arbitrary number of Records without either blocking or allocation.
type asyncHandler struct {
	ringBuffer [100]slog.Record
	next       int
}

func newAsyncHandler() *asyncHandler {
	return &asyncHandler{}
}

func (*asyncHandler) Enabled(context.Context, slog.Level) bool { return true }

func (h *asyncHandler) Handle(_ context.Context, r slog.Record) error {
	h.ringBuffer[h.next] = r.Clone()
	h.next = (h.next + 1) % len(h.ringBuffer)
	return nil
}

func (*asyncHandler) WithAttrs([]slog.Attr) slog.Handler {
	panic("asyncHandler: With unimplemented")
}

func (*asyncHandler) WithGroup(string) slog.Handler {
	panic("asyncHandler: WithGroup unimplemented")
}

// A disabledHandler's Enabled method always returns false.
type disabledHandler struct{}

func (disabledHandler) Enabled(context.Context, slog.Level) bool  { return false }
func (disabledHandler) Handle(context.Context, slog.Record) error { panic("should not be called") }

func (disabledHandler) WithAttrs([]slog.Attr) slog.Handler {
	panic("disabledHandler: With unimplemented")
}

func (disabledHandler) WithGroup(string) slog.Handler {
	panic("disabledHandler: WithGroup unimplemented")
}
