// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package telemetry

import (
	"fmt"
	"time"
)

type SpanContext struct {
	TraceID TraceID
	SpanID  SpanID
}

type Span struct {
	Name     string
	ID       SpanContext
	ParentID SpanID
	Start    time.Time
	Finish   time.Time
	Tags     TagList
	Events   []Event
}

func (s *SpanContext) Format(f fmt.State, r rune) {
	fmt.Fprintf(f, "%v:%v", s.TraceID, s.SpanID)
}

func (s *Span) Format(f fmt.State, r rune) {
	fmt.Fprintf(f, "%v %v", s.Name, s.ID)
	if s.ParentID.IsValid() {
		fmt.Fprintf(f, "[%v]", s.ParentID)
	}
	fmt.Fprintf(f, " %v->%v", s.Start, s.Finish)
}
