// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ocagent adds the ability to export all telemetry to an ocagent.
// This keeps the compile time dependencies to zero and allows the agent to
// have the exporters needed for telemetry aggregation and viewing systems.
package ocagent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"golang.org/x/tools/internal/telemetry/event"
	"golang.org/x/tools/internal/telemetry/export"
	"golang.org/x/tools/internal/telemetry/export/metric"
	"golang.org/x/tools/internal/telemetry/export/ocagent/wire"
)

type Config struct {
	Start   time.Time
	Host    string
	Process uint32
	Client  *http.Client
	Service string
	Address string
	Rate    time.Duration
}

// Discover finds the local agent to export to, it will return nil if there
// is not one running.
// TODO: Actually implement a discovery protocol rather than a hard coded address
func Discover() *Config {
	return &Config{
		Address: "http://localhost:55678",
	}
}

type Exporter struct {
	mu      sync.Mutex
	config  Config
	spans   []*export.Span
	metrics []metric.Data
}

// Connect creates a process specific exporter with the specified
// serviceName and the address of the ocagent to which it will upload
// its telemetry.
func Connect(config *Config) *Exporter {
	if config == nil || config.Address == "off" {
		return nil
	}
	exporter := &Exporter{config: *config}
	if exporter.config.Start.IsZero() {
		exporter.config.Start = time.Now()
	}
	if exporter.config.Host == "" {
		hostname, _ := os.Hostname()
		exporter.config.Host = hostname
	}
	if exporter.config.Process == 0 {
		exporter.config.Process = uint32(os.Getpid())
	}
	if exporter.config.Client == nil {
		exporter.config.Client = http.DefaultClient
	}
	if exporter.config.Service == "" {
		exporter.config.Service = filepath.Base(os.Args[0])
	}
	if exporter.config.Rate == 0 {
		exporter.config.Rate = 2 * time.Second
	}
	go func() {
		for range time.Tick(exporter.config.Rate) {
			exporter.Flush()
		}
	}()
	return exporter
}

func (e *Exporter) ProcessEvent(ctx context.Context, ev event.Event, tagMap event.TagMap) context.Context {
	switch {
	case ev.IsEndSpan():
		e.mu.Lock()
		defer e.mu.Unlock()
		span := export.GetSpan(ctx)
		if span != nil {
			e.spans = append(e.spans, span)
		}
	case ev.IsRecord():
		e.mu.Lock()
		defer e.mu.Unlock()
		data := metric.Entries.Get(tagMap).([]metric.Data)
		e.metrics = append(e.metrics, data...)
	}
	return ctx
}

func (e *Exporter) Flush() {
	e.mu.Lock()
	defer e.mu.Unlock()
	spans := make([]*wire.Span, len(e.spans))
	for i, s := range e.spans {
		spans[i] = convertSpan(s)
	}
	e.spans = nil
	metrics := make([]*wire.Metric, len(e.metrics))
	for i, m := range e.metrics {
		metrics[i] = convertMetric(m, e.config.Start)
	}
	e.metrics = nil

	if len(spans) > 0 {
		e.send("/v1/trace", &wire.ExportTraceServiceRequest{
			Node:  e.config.buildNode(),
			Spans: spans,
			//TODO: Resource?
		})
	}
	if len(metrics) > 0 {
		e.send("/v1/metrics", &wire.ExportMetricsServiceRequest{
			Node:    e.config.buildNode(),
			Metrics: metrics,
			//TODO: Resource?
		})
	}
}

func (cfg *Config) buildNode() *wire.Node {
	return &wire.Node{
		Identifier: &wire.ProcessIdentifier{
			HostName:       cfg.Host,
			Pid:            cfg.Process,
			StartTimestamp: convertTimestamp(cfg.Start),
		},
		LibraryInfo: &wire.LibraryInfo{
			Language:           wire.LanguageGo,
			ExporterVersion:    "0.0.1",
			CoreLibraryVersion: "x/tools",
		},
		ServiceInfo: &wire.ServiceInfo{
			Name: cfg.Service,
		},
	}
}

func (e *Exporter) send(endpoint string, message interface{}) {
	blob, err := json.Marshal(message)
	if err != nil {
		errorInExport("ocagent failed to marshal message for %v: %v", endpoint, err)
		return
	}
	uri := e.config.Address + endpoint
	req, err := http.NewRequest("POST", uri, bytes.NewReader(blob))
	if err != nil {
		errorInExport("ocagent failed to build request for %v: %v", uri, err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := e.config.Client.Do(req)
	if err != nil {
		errorInExport("ocagent failed to send message: %v \n", err)
		return
	}
	if res.Body != nil {
		res.Body.Close()
	}
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

func convertSpan(span *export.Span) *wire.Span {
	result := &wire.Span{
		TraceID:                 span.ID.TraceID[:],
		SpanID:                  span.ID.SpanID[:],
		TraceState:              nil, //TODO?
		ParentSpanID:            span.ParentID[:],
		Name:                    toTruncatableString(span.Name),
		Kind:                    wire.UnspecifiedSpanKind,
		StartTime:               convertTimestamp(span.Start().At),
		EndTime:                 convertTimestamp(span.Finish().At),
		Attributes:              convertAttributes(event.Filter(span.Start().Tags(), event.Name)),
		TimeEvents:              convertEvents(span.Events()),
		SameProcessAsParentSpan: true,
		//TODO: StackTrace?
		//TODO: Links?
		//TODO: Status?
		//TODO: Resource?
	}
	return result
}

func convertMetric(data metric.Data, start time.Time) *wire.Metric {
	descriptor := dataToMetricDescriptor(data)
	timeseries := dataToTimeseries(data, start)

	if descriptor == nil && timeseries == nil {
		return nil
	}

	// TODO: handle Histogram metrics
	return &wire.Metric{
		MetricDescriptor: descriptor,
		Timeseries:       timeseries,
		// TODO: attach Resource?
	}
}

func convertAttributes(it event.TagIterator) *wire.Attributes {
	if !it.Valid() {
		return nil
	}
	attributes := make(map[string]wire.Attribute)
	for ; it.Valid(); it.Advance() {
		tag := it.Tag()
		attributes[tag.Key.Name()] = convertAttribute(tag)
	}
	return &wire.Attributes{AttributeMap: attributes}
}

func convertAttribute(tag event.Tag) wire.Attribute {
	switch key := tag.Key.(type) {
	case *event.IntKey:
		return wire.IntAttribute{IntValue: int64(key.From(tag))}
	case *event.Int8Key:
		return wire.IntAttribute{IntValue: int64(key.From(tag))}
	case *event.Int16Key:
		return wire.IntAttribute{IntValue: int64(key.From(tag))}
	case *event.Int32Key:
		return wire.IntAttribute{IntValue: int64(key.From(tag))}
	case *event.Int64Key:
		return wire.IntAttribute{IntValue: int64(key.From(tag))}
	case *event.UIntKey:
		return wire.IntAttribute{IntValue: int64(key.From(tag))}
	case *event.UInt8Key:
		return wire.IntAttribute{IntValue: int64(key.From(tag))}
	case *event.UInt16Key:
		return wire.IntAttribute{IntValue: int64(key.From(tag))}
	case *event.UInt32Key:
		return wire.IntAttribute{IntValue: int64(key.From(tag))}
	case *event.UInt64Key:
		return wire.IntAttribute{IntValue: int64(key.From(tag))}
	case *event.Float32Key:
		return wire.DoubleAttribute{DoubleValue: float64(key.From(tag))}
	case *event.Float64Key:
		return wire.DoubleAttribute{DoubleValue: key.From(tag)}
	case *event.BooleanKey:
		return wire.BoolAttribute{BoolValue: key.From(tag)}
	case *event.StringKey:
		return wire.StringAttribute{StringValue: toTruncatableString(key.From(tag))}
	case *event.ErrorKey:
		return wire.StringAttribute{StringValue: toTruncatableString(key.From(tag).Error())}
	case *event.ValueKey:
		return wire.StringAttribute{StringValue: toTruncatableString(fmt.Sprint(key.From(tag)))}
	default:
		return wire.StringAttribute{StringValue: toTruncatableString(fmt.Sprintf("%T", key))}
	}
}

func convertEvents(events []event.Event) *wire.TimeEvents {
	//TODO: MessageEvents?
	result := make([]wire.TimeEvent, len(events))
	for i, event := range events {
		result[i] = convertEvent(event)
	}
	return &wire.TimeEvents{TimeEvent: result}
}

func convertEvent(ev event.Event) wire.TimeEvent {
	return wire.TimeEvent{
		Time:       convertTimestamp(ev.At),
		Annotation: convertAnnotation(ev),
	}
}

func convertAnnotation(ev event.Event) *wire.Annotation {
	tags := ev.Tags()
	if !tags.Valid() {
		return nil
	}
	tagMap := event.TagMap(ev)
	description := event.Msg.Get(tagMap)
	tags = event.Filter(tags, event.Msg)
	if description == "" {
		err := event.Err.Get(tagMap)
		tags = event.Filter(tags, event.Err)
		if err != nil {
			description = err.Error()
		}
	}
	return &wire.Annotation{
		Description: toTruncatableString(description),
		Attributes:  convertAttributes(tags),
	}
}
