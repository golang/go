// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package benchmarks

import (
	"bytes"
	"context"
	"log/slog"
	"slices"
	"testing"
)

func TestHandlers(t *testing.T) {
	ctx := context.Background()
	r := slog.NewRecord(testTime, slog.LevelInfo, testMessage, 0)
	r.AddAttrs(testAttrs...)
	t.Run("text", func { t ->
		var b bytes.Buffer
		h := newFastTextHandler(&b)
		if err := h.Handle(ctx, r); err != nil {
			t.Fatal(err)
		}
		got := b.String()
		if got != wantText {
			t.Errorf("\ngot  %q\nwant %q", got, wantText)
		}
	})
	t.Run("async", func { t ->
		h := newAsyncHandler()
		if err := h.Handle(ctx, r); err != nil {
			t.Fatal(err)
		}
		got := h.ringBuffer[0]
		if !got.Time.Equal(r.Time) || !slices.EqualFunc(attrSlice(got), attrSlice(r), slog.Attr.Equal) {
			t.Errorf("got %+v, want %+v", got, r)
		}
	})
}

func attrSlice(r slog.Record) []slog.Attr {
	var as []slog.Attr
	r.Attrs(func { a ->
		as = append(as, a)
		return true
	})
	return as
}
