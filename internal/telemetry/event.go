// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package telemetry

import "golang.org/x/tools/internal/telemetry/event"

type Event = event.Event
type Tag = event.Tag
type TagList = event.TagList
type MetricData = event.MetricData

const (
	EventLog       = event.LogType
	EventStartSpan = event.StartSpanType
	EventEndSpan   = event.EndSpanType
	EventLabel     = event.LabelType
	EventDetach    = event.DetachType
)
