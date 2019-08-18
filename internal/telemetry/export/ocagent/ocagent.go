// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ocagent adds the ability to export all telemetry to an ocagent.
// This keeps the complie time dependencies to zero and allows the agent to
// have the exporters needed for telemetry aggregation and viewing systems.
package ocagent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sync"
	"time"

	"golang.org/x/tools/internal/telemetry"
	"golang.org/x/tools/internal/telemetry/export"
	"golang.org/x/tools/internal/telemetry/export/ocagent/wire"
	"golang.org/x/tools/internal/telemetry/tag"
)

const DefaultAddress = "http://localhost:55678"
const exportRate = 2 * time.Second

type exporter struct {
	mu      sync.Mutex
	address string
	node    *wire.Node
	spans   []*wire.Span
	metrics []*wire.Metric
}

// Connect creates a process specific exporter with the specified
// serviceName and the address of the ocagent to which it will upload
// its telemetry.
func Connect(service, address string) export.Exporter {
	if address == "off" {
		return nil
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
	go func() {
		for _ = range time.Tick(exportRate) {
			exporter.flush()
		}
	}()
	return exporter
}

func (e *exporter) StartSpan(ctx context.Context, span *telemetry.Span) {}

func (e *exporter) FinishSpan(ctx context.Context, span *telemetry.Span) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.spans = append(e.spans, convertSpan(span))
}

func (e *exporter) Log(context.Context, telemetry.Event) {}

func (e *exporter) Metric(ctx context.Context, data telemetry.MetricData) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.metrics = append(e.metrics, convertMetric(data))
}

func (e *exporter) flush() {
	e.mu.Lock()
	defer e.mu.Unlock()
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
	if s == "" {
		return nil
	}
	return &wire.TruncatableString{Value: s}
}

func convertSpan(span *telemetry.Span) *wire.Span {
	result := &wire.Span{
		TraceId:                 span.ID.TraceID[:],
		SpanId:                  span.ID.SpanID[:],
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

func convertMetric(data telemetry.MetricData) *wire.Metric {
	return nil //TODO:
}

func convertAttributes(tags telemetry.TagList) *wire.Attributes {
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
	case int:
		return wire.IntAttribute{IntValue: int64(v)}
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

func convertEvents(events []telemetry.Event) *wire.TimeEvents {
	//TODO: MessageEvents?
	result := make([]wire.TimeEvent, len(events))
	for i, event := range events {
		result[i] = convertEvent(event)
	}
	return &wire.TimeEvents{TimeEvent: result}
}

func convertEvent(event telemetry.Event) wire.TimeEvent {
	return wire.TimeEvent{
		Time:       convertTimestamp(event.At),
		Annotation: convertAnnotation(event),
	}
}

func convertAnnotation(event telemetry.Event) *wire.Annotation {
	description := event.Message
	if description == "" && event.Error != nil {
		description = event.Error.Error()
		event.Error = nil
	}
	tags := event.Tags
	if event.Error != nil {
		tags = append(tags, tag.Of("Error", event.Error))
	}
	if description == "" && len(tags) == 0 {
		return nil
	}
	return &wire.Annotation{
		Description: toTruncatableString(description),
		Attributes:  convertAttributes(tags),
	}
}
