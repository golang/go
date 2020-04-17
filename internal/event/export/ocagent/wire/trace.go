// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wire

type ExportTraceServiceRequest struct {
	Node     *Node     `json:"node,omitempty"`
	Spans    []*Span   `json:"spans,omitempty"`
	Resource *Resource `json:"resource,omitempty"`
}

type Span struct {
	TraceID                 []byte             `json:"trace_id,omitempty"`
	SpanID                  []byte             `json:"span_id,omitempty"`
	TraceState              *TraceState        `json:"tracestate,omitempty"`
	ParentSpanID            []byte             `json:"parent_span_id,omitempty"`
	Name                    *TruncatableString `json:"name,omitempty"`
	Kind                    SpanKind           `json:"kind,omitempty"`
	StartTime               Timestamp          `json:"start_time,omitempty"`
	EndTime                 Timestamp          `json:"end_time,omitempty"`
	Attributes              *Attributes        `json:"attributes,omitempty"`
	StackTrace              *StackTrace        `json:"stack_trace,omitempty"`
	TimeEvents              *TimeEvents        `json:"time_events,omitempty"`
	Links                   *Links             `json:"links,omitempty"`
	Status                  *Status            `json:"status,omitempty"`
	Resource                *Resource          `json:"resource,omitempty"`
	SameProcessAsParentSpan bool               `json:"same_process_as_parent_span,omitempty"`
	ChildSpanCount          bool               `json:"child_span_count,omitempty"`
}

type TraceState struct {
	Entries []*TraceStateEntry `json:"entries,omitempty"`
}

type TraceStateEntry struct {
	Key   string `json:"key,omitempty"`
	Value string `json:"value,omitempty"`
}

type SpanKind int32

const (
	UnspecifiedSpanKind SpanKind = 0
	ServerSpanKind      SpanKind = 1
	ClientSpanKind      SpanKind = 2
)

type TimeEvents struct {
	TimeEvent                 []TimeEvent `json:"timeEvent,omitempty"`
	DroppedAnnotationsCount   int32       `json:"dropped_annotations_count,omitempty"`
	DroppedMessageEventsCount int32       `json:"dropped_message_events_count,omitempty"`
}

type TimeEvent struct {
	Time         Timestamp     `json:"time,omitempty"`
	MessageEvent *MessageEvent `json:"messageEvent,omitempty"`
	Annotation   *Annotation   `json:"annotation,omitempty"`
}

type Annotation struct {
	Description *TruncatableString `json:"description,omitempty"`
	Attributes  *Attributes        `json:"attributes,omitempty"`
}

type MessageEvent struct {
	Type             MessageEventType `json:"type,omitempty"`
	ID               uint64           `json:"id,omitempty"`
	UncompressedSize uint64           `json:"uncompressed_size,omitempty"`
	CompressedSize   uint64           `json:"compressed_size,omitempty"`
}

type MessageEventType int32

const (
	UnspecifiedMessageEvent MessageEventType = iota
	SentMessageEvent
	ReceivedMessageEvent
)

type TimeEventValue interface {
	tagTimeEventValue()
}

func (Annotation) tagTimeEventValue()   {}
func (MessageEvent) tagTimeEventValue() {}

type Links struct {
	Link              []*Link `json:"link,omitempty"`
	DroppedLinksCount int32   `json:"dropped_links_count,omitempty"`
}

type Link struct {
	TraceID    []byte      `json:"trace_id,omitempty"`
	SpanID     []byte      `json:"span_id,omitempty"`
	Type       LinkType    `json:"type,omitempty"`
	Attributes *Attributes `json:"attributes,omitempty"`
	TraceState *TraceState `json:"tracestate,omitempty"`
}

type LinkType int32

const (
	UnspecifiedLinkType LinkType = 0
	ChildLinkType       LinkType = 1
	ParentLinkType      LinkType = 2
)

type Status struct {
	Code    int32  `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
}
