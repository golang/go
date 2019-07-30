// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ocagent adds the ability to export all telemetry to an ocagent.
// This keeps the complie time dependencies to zero and allows the agent to
// have the exporters needed for telemetry aggregation and viewing systems.
package ocagent

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"

	"golang.org/x/tools/internal/lsp/telemetry/log"
	"golang.org/x/tools/internal/lsp/telemetry/metric"
	"golang.org/x/tools/internal/lsp/telemetry/ocagent/wire"
	"golang.org/x/tools/internal/lsp/telemetry/tag"
	"golang.org/x/tools/internal/lsp/telemetry/trace"
	"golang.org/x/tools/internal/lsp/telemetry/worker"
)

const DefaultAddress = "http://localhost:55678"
const exportRate = 2 * time.Second

type exporter struct {
	address string
	node    *wire.Node
	spans   []*wire.Span
	metrics []*wire.Metric
}

func Export(service, address string) {
	if address == "off" {
		return
	}
	hostname, _ := os.Hostname()
	exporter := &exporter{
		address: address,
		node: &wire.Node{
			Identifier: &wire.ProcessIdentifier{
				HostName:       hostname,
				Pid:            uint32(os.Getpid()),
				StartTimestamp: convertTimestamp(time.Now()),
			},
			LibraryInfo: &wire.LibraryInfo{
				Language:           wire.LanguageGo,
				ExporterVersion:    "0.0.1",
				CoreLibraryVersion: "x/tools",
			},
			ServiceInfo: &wire.ServiceInfo{
				Name: service,
			},
		},
	}
	if exporter.address == "" {
		exporter.address = DefaultAddress
	}
	//TODO: add metrics once the ocagent json metric interface works
	trace.RegisterObservers(exporter.observeTrace)
	go func() {
		for _ = range time.Tick(exportRate) {
			worker.Do(func() {
				exporter.flush()
			})
		}
	}()
}

func (e *exporter) observeTrace(span *trace.Span) {
	// is this a completed span?
	if span.Finish.IsZero() {
		return
	}
	e.spans = append(e.spans, convertSpan(span))
}

func (e *exporter) observeMetric(data metric.Data) {
	e.metrics = append(e.metrics, convertMetric(data))
}

func (e *exporter) flush() {
	spans := e.spans
	e.spans = nil
	metrics := e.metrics
	e.metrics = nil

	if len(spans) > 0 {
		e.send("/v1/trace", &wire.ExportTraceServiceRequest{
			Node:  e.node,
			Spans: spans,
			//TODO: Resource?
		})
	}
	if len(metrics) > 0 {
		e.send("/v1/metrics", &wire.ExportMetricsServiceRequest{
			Node:    e.node,
			Metrics: metrics,
			//TODO: Resource?
		})
	}
}

func (e *exporter) send(endpoint string, message interface{}) {
	blob, err := json.Marshal(message)
	if err != nil {
		errorInExport("ocagent failed to marshal message for %v: %v", endpoint, err)
		return
	}
	uri := e.address + endpoint
	req, err := http.NewRequest("POST", uri, bytes.NewReader(blob))
	if err != nil {
		errorInExport("ocagent failed to build request for %v: %v", uri, err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		errorInExport("ocagent failed to send message: %v \n", err)
		return
	}
	res.Body.Close()
	return
}

func errorInExport(message string, args ...interface{}) {
	// This function is useful when debugging the exporter, but in general we
	// want to just drop any export
}

func convertTimestamp(t time.Time) wire.Timestamp {
	return t.Format(time.RFC3339Nano)
}

func toTruncatableString(s string) *wire.TruncatableString {
	return &wire.TruncatableString{Value: s}
}

func convertSpan(span *trace.Span) *wire.Span {
	result := &wire.Span{
		TraceId:                 span.TraceID[:],
		SpanId:                  span.SpanID[:],
		TraceState:              nil, //TODO?
		ParentSpanId:            span.ParentID[:],
		Name:                    toTruncatableString(span.Name),
		Kind:                    wire.UnspecifiedSpanKind,
		StartTime:               convertTimestamp(span.Start),
		EndTime:                 convertTimestamp(span.Finish),
		Attributes:              convertAttributes(span.Tags),
		TimeEvents:              convertEvents(span.Events),
		SameProcessAsParentSpan: true,
		//TODO: StackTrace?
		//TODO: Links?
		//TODO: Status?
		//TODO: Resource?
	}
	return result
}

func convertMetric(data metric.Data) *wire.Metric {
	return nil //TODO:
}

func convertAttributes(tags tag.List) *wire.Attributes {
	if len(tags) == 0 {
		return nil
	}
	attributes := make(map[string]wire.Attribute)
	for _, tag := range tags {
		attributes[fmt.Sprint(tag.Key)] = convertAttribute(tag.Value)
	}
	return &wire.Attributes{AttributeMap: attributes}
}

func convertAttribute(v interface{}) wire.Attribute {
	switch v := v.(type) {
	case int8:
		return wire.IntAttribute{IntValue: int64(v)}
	case int16:
		return wire.IntAttribute{IntValue: int64(v)}
	case int32:
		return wire.IntAttribute{IntValue: int64(v)}
	case int64:
		return wire.IntAttribute{IntValue: v}
	case uint8:
		return wire.IntAttribute{IntValue: int64(v)}
	case uint16:
		return wire.IntAttribute{IntValue: int64(v)}
	case uint32:
		return wire.IntAttribute{IntValue: int64(v)}
	case uint64:
		return wire.IntAttribute{IntValue: int64(v)}
	case uint:
		return wire.IntAttribute{IntValue: int64(v)}
	case float32:
		return wire.DoubleAttribute{DoubleValue: float64(v)}
	case float64:
		return wire.DoubleAttribute{DoubleValue: v}
	case bool:
		return wire.BoolAttribute{BoolValue: v}
	case string:
		return wire.StringAttribute{StringValue: toTruncatableString(v)}
	default:
		return wire.StringAttribute{StringValue: toTruncatableString(fmt.Sprint(v))}
	}
}

func convertEvents(events []trace.Event) *wire.TimeEvents {
	//TODO: MessageEvents?
	result := make([]wire.TimeEvent, len(events))
	for i, event := range events {
		result[i] = convertEvent(event)
	}
	return &wire.TimeEvents{TimeEvent: result}
}

func convertEvent(event trace.Event) wire.TimeEvent {
	return wire.TimeEvent{
		Time:       convertTimestamp(event.Time),
		Annotation: convertAnnotation(event.Tags),
	}
}

func convertAnnotation(tags tag.List) *wire.Annotation {
	entry := log.ToEntry(nil, time.Time{}, tags)
	description := entry.Message
	if description == "" && entry.Error != nil {
		description = entry.Error.Error()
		entry.Error = nil
	}
	tags = entry.Tags
	if entry.Error != nil {
		tags = append(tags, tag.Of("Error", entry.Error))
	}
	return &wire.Annotation{
		Description: toTruncatableString(description),
		Attributes:  convertAttributes(tags),
	}
}
