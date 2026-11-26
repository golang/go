// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package traceviewer

import (
	"internal/trace/traceviewer/format"
	"testing"
	"time"
)

func TestSliceTimeRange(t *testing.T) {
	for _, tt := range []struct {
		name       string
		start, end time.Duration
		want       bool
	}{
		{"before", 1, 9, false},
		{"touches start", 1, 10, true},
		{"inside", 12, 18, true},
		{"encloses", 5, 25, true},
		{"touches end", 20, 29, true},
		{"after", 21, 29, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var events []*format.Event
			e := NewEmitter(TraceConsumer{
				ConsumeTimeUnit: func(string) {},
				ConsumeViewerEvent: func(ev *format.Event, _ bool) {
					events = append(events, ev)
				},
			}, 10, 20)
			s := SliceEvent{Ts: tt.start, Dur: tt.end - tt.start}
			e.Slice(s)
			e.AsyncSlice(AsyncSliceEvent{SliceEvent: s})
			if !tt.want {
				if len(events) != 0 {
					t.Errorf("emitted %d events for a disjoint slice", len(events))
				}
				return
			}
			if len(events) != 3 {
				t.Fatalf("emitted %d events, want a slice and an async begin/end pair", len(events))
			}
			if events[0].Time != viewerTime(tt.start) || events[0].Dur != viewerTime(tt.end-tt.start) ||
				events[1].Time != viewerTime(tt.start) || events[2].Time != viewerTime(tt.end) {
				t.Error("emitted slice lost its original endpoints")
			}
		})
	}
}
