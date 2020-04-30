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

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/core"
	"golang.org/x/tools/internal/event/export"
	"golang.org/x/tools/internal/event/export/metric"
	"golang.org/x/tools/internal/event/export/ocagent/wire"
	"golang.org/x/tools/internal/event/keys"
	"golang.org/x/tools/internal/event/label"
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

var (
	connectMu sync.Mutex
	exporters = make(map[Config]*Exporter)
)

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
	resolved := *config
	if resolved.Host == "" {
		hostname, _ := os.Hostname()
		resolved.Host = hostname
	}
	if resolved.Process == 0 {
		resolved.Process = uint32(os.Getpid())
	}
	if resolved.Client == nil {
		resolved.Client = http.DefaultClient
	}
	if resolved.Service == "" {
		resolved.Service = filepath.Base(os.Args[0])
	}
	if resolved.Rate == 0 {
		resolved.Rate = 2 * time.Second
	}

	connectMu.Lock()
	defer connectMu.Unlock()
	if exporter, found := exporters[resolved]; found {
		return exporter
	}
	exporter := &Exporter{config: resolved}
	exporters[resolved] = exporter
	if exporter.config.Start.IsZero() {
		exporter.config.Start = time.Now()
	}
	go func() {
		for range time.Tick(exporter.config.Rate) {
			exporter.Flush()
		}
	}()
	return exporter
}

func (e *Exporter) ProcessEvent(ctx context.Context, ev core.Event, lm label.Map) context.Context {
	switch {
	case event.IsEnd(ev):
		e.mu.Lock()
		defer e.mu.Unlock()
		span := export.GetSpan(ctx)
		if span != nil {
			e.spans = append(e.spans, span)
		}
	case event.IsMetric(ev):
		e.mu.Lock()
		defer e.mu.Unlock()
		data := metric.Entries.Get(lm).([]metric.Data)
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
		StartTime:               convertTimestamp(span.Start().At()),
		EndTime:                 convertTimestamp(span.Finish().At()),
		Attributes:              convertAttributes(span.Start(), 1),
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

func skipToValidLabel(list label.List, index int) (int, label.Label) {
	// skip to the first valid label
	for ; list.Valid(index); index++ {
		l := list.Label(index)
		if !l.Valid() || l.Key() == keys.Label {
			continue
		}
		return index, l
	}
	return -1, label.Label{}
}

func convertAttributes(list label.List, index int) *wire.Attributes {
	index, l := skipToValidLabel(list, index)
	if !l.Valid() {
		return nil
	}
	attributes := make(map[string]wire.Attribute)
	for {
		if l.Valid() {
			attributes[l.Key().Name()] = convertAttribute(l)
		}
		index++
		if !list.Valid(index) {
			return &wire.Attributes{AttributeMap: attributes}
		}
		l = list.Label(index)
	}
}

func convertAttribute(l label.Label) wire.Attribute {
	switch key := l.Key().(type) {
	case *keys.Int:
		return wire.IntAttribute{IntValue: int64(key.From(l))}
	case *keys.Int8:
		return wire.IntAttribute{IntValue: int64(key.From(l))}
	case *keys.Int16:
		return wire.IntAttribute{IntValue: int64(key.From(l))}
	case *keys.Int32:
		return wire.IntAttribute{IntValue: int64(key.From(l))}
	case *keys.Int64:
		return wire.IntAttribute{IntValue: int64(key.From(l))}
	case *keys.UInt:
		return wire.IntAttribute{IntValue: int64(key.From(l))}
	case *keys.UInt8:
		return wire.IntAttribute{IntValue: int64(key.From(l))}
	case *keys.UInt16:
		return wire.IntAttribute{IntValue: int64(key.From(l))}
	case *keys.UInt32:
		return wire.IntAttribute{IntValue: int64(key.From(l))}
	case *keys.UInt64:
		return wire.IntAttribute{IntValue: int64(key.From(l))}
	case *keys.Float32:
		return wire.DoubleAttribute{DoubleValue: float64(key.From(l))}
	case *keys.Float64:
		return wire.DoubleAttribute{DoubleValue: key.From(l)}
	case *keys.Boolean:
		return wire.BoolAttribute{BoolValue: key.From(l)}
	case *keys.String:
		return wire.StringAttribute{StringValue: toTruncatableString(key.From(l))}
	case *keys.Error:
		return wire.StringAttribute{StringValue: toTruncatableString(key.From(l).Error())}
	case *keys.Value:
		return wire.StringAttribute{StringValue: toTruncatableString(fmt.Sprint(key.From(l)))}
	default:
		return wire.StringAttribute{StringValue: toTruncatableString(fmt.Sprintf("%T", key))}
	}
}

func convertEvents(events []core.Event) *wire.TimeEvents {
	//TODO: MessageEvents?
	result := make([]wire.TimeEvent, len(events))
	for i, event := range events {
		result[i] = convertEvent(event)
	}
	return &wire.TimeEvents{TimeEvent: result}
}

func convertEvent(ev core.Event) wire.TimeEvent {
	return wire.TimeEvent{
		Time:       convertTimestamp(ev.At()),
		Annotation: convertAnnotation(ev),
	}
}

func getAnnotationDescription(ev core.Event) (string, int) {
	l := ev.Label(0)
	if l.Key() != keys.Msg {
		return "", 0
	}
	if msg := keys.Msg.From(l); msg != "" {
		return msg, 1
	}
	l = ev.Label(1)
	if l.Key() != keys.Err {
		return "", 1
	}
	if err := keys.Err.From(l); err != nil {
		return err.Error(), 2
	}
	return "", 2
}

func convertAnnotation(ev core.Event) *wire.Annotation {
	description, index := getAnnotationDescription(ev)
	if _, l := skipToValidLabel(ev, index); !l.Valid() && description == "" {
		return nil
	}
	return &wire.Annotation{
		Description: toTruncatableString(description),
		Attributes:  convertAttributes(ev, index),
	}
}
