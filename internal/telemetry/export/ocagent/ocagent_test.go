// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ocagent_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"sync"
	"testing"
	"time"

	"golang.org/x/tools/internal/telemetry/event"
	"golang.org/x/tools/internal/telemetry/export"
	"golang.org/x/tools/internal/telemetry/export/ocagent"
	"golang.org/x/tools/internal/telemetry/metric"
)

const testNodeStr = `{
	"node":{
		"identifier":{
			"host_name":"tester",
			"pid":1,
			"start_timestamp":"1970-01-01T00:00:00Z"
		},
		"library_info":{
			"language":4,
			"exporter_version":"0.0.1",
			"core_library_version":"x/tools"
		},
		"service_info":{
			"name":"ocagent-tests"
		}
	},`

var (
	keyDB    = event.NewStringKey("db", "the database name")
	keyHello = event.NewStringKey("hello", "a metric grouping key")
	keyWorld = event.NewStringKey("world", "another metric grouping key")

	key1DB = event.NewStringKey("1_db", "A test string key")

	key2aAge      = event.NewFloat64Key("2a_age", "A test float64 key")
	key2bTTL      = event.NewFloat32Key("2b_ttl", "A test float32 key")
	key2cExpiryMS = event.NewFloat64Key("2c_expiry_ms", "A test float64 key")

	key3aRetry = event.NewBooleanKey("3a_retry", "A test boolean key")
	key3bStale = event.NewBooleanKey("3b_stale", "Another test boolean key")

	key4aMax      = event.NewIntKey("4a_max", "A test int key")
	key4bOpcode   = event.NewInt8Key("4b_opcode", "A test int8 key")
	key4cBase     = event.NewInt16Key("4c_base", "A test int16 key")
	key4eChecksum = event.NewInt32Key("4e_checksum", "A test int32 key")
	key4fMode     = event.NewInt64Key("4f_mode", "A test int64 key")

	key5aMin     = event.NewUIntKey("5a_min", "A test uint key")
	key5bMix     = event.NewUInt8Key("5b_mix", "A test uint8 key")
	key5cPort    = event.NewUInt16Key("5c_port", "A test uint16 key")
	key5dMinHops = event.NewUInt32Key("5d_min_hops", "A test uint32 key")
	key5eMaxHops = event.NewUInt64Key("5e_max_hops", "A test uint64 key")
)

type testExporter struct {
	ocagent *ocagent.Exporter
	sent    fakeSender
	start   time.Time
	at      time.Time
	end     time.Time
}

func registerExporter() *testExporter {
	exporter := &testExporter{}
	cfg := ocagent.Config{
		Host:    "tester",
		Process: 1,
		Service: "ocagent-tests",
		Client:  &http.Client{Transport: &exporter.sent},
	}
	cfg.Start, _ = time.Parse(time.RFC3339Nano, "1970-01-01T00:00:00Z")
	exporter.ocagent = ocagent.Connect(&cfg)
	exporter.start, _ = time.Parse(time.RFC3339Nano, "1970-01-01T00:00:30Z")
	exporter.at, _ = time.Parse(time.RFC3339Nano, "1970-01-01T00:00:40Z")
	exporter.end, _ = time.Parse(time.RFC3339Nano, "1970-01-01T00:00:50Z")
	event.SetExporter(exporter)
	return exporter
}

func (e *testExporter) ProcessEvent(ctx context.Context, ev event.Event) (context.Context, event.Event) {
	switch {
	case ev.IsStartSpan():
		ev.At = e.start
	case ev.IsEndSpan():
		ev.At = e.end
	default:
		ev.At = e.at
	}
	ctx, ev = export.Tag(ctx, ev)
	ctx, ev = export.ContextSpan(ctx, ev)
	ctx, ev = e.ocagent.ProcessEvent(ctx, ev)
	if ev.IsStartSpan() {
		span := export.GetSpan(ctx)
		span.ID = export.SpanContext{}
	}
	return ctx, ev
}

func (e *testExporter) Metric(ctx context.Context, data event.MetricData) {
	switch data := data.(type) {
	case *metric.Int64Data:
		data.EndTime = &e.start
	}
	e.ocagent.Metric(ctx, data)
}

func (e *testExporter) Output(route string) []byte {
	e.ocagent.Flush()
	return e.sent.get(route)
}

func checkJSON(t *testing.T, got, want []byte) {
	// compare the compact form, to allow for formatting differences
	g := &bytes.Buffer{}
	if err := json.Compact(g, got); err != nil {
		t.Fatal(err)
	}
	w := &bytes.Buffer{}
	if err := json.Compact(w, want); err != nil {
		t.Fatal(err)
	}
	if g.String() != w.String() {
		t.Fatalf("Got:\n%s\nWant:\n%s", g, w)
	}
}

type fakeSender struct {
	mu   sync.Mutex
	data map[string][]byte
}

func (s *fakeSender) get(route string) []byte {
	s.mu.Lock()
	defer s.mu.Unlock()
	data, found := s.data[route]
	if found {
		delete(s.data, route)
	}
	return data
}

func (s *fakeSender) RoundTrip(req *http.Request) (*http.Response, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.data == nil {
		s.data = make(map[string][]byte)
	}
	data, err := ioutil.ReadAll(req.Body)
	if err != nil {
		return nil, err
	}
	path := req.URL.EscapedPath()
	if _, found := s.data[path]; found {
		return nil, fmt.Errorf("duplicate delivery to %v", path)
	}
	s.data[path] = data
	return &http.Response{
		Status:     "200 OK",
		StatusCode: 200,
		Proto:      "HTTP/1.0",
		ProtoMajor: 1,
		ProtoMinor: 0,
	}, nil
}
