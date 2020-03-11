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
	keyDB    = &event.Key{Name: "db"}
	keyHello = &event.Key{Name: "hello"}
	keyWorld = &event.Key{Name: "world"}

	key1DB = &event.Key{Name: "1_db"}

	key2aAge      = &event.Key{Name: "2a_age"}
	key2bTTL      = &event.Key{Name: "2b_ttl"}
	key2cExpiryMS = &event.Key{Name: "2c_expiry_ms"}

	key3aRetry = &event.Key{Name: "3a_retry"}
	key3bStale = &event.Key{Name: "3b_stale"}

	key4aMax      = &event.Key{Name: "4a_max"}
	key4bOpcode   = &event.Key{Name: "4b_opcode"}
	key4cBase     = &event.Key{Name: "4c_base"}
	key4eChecksum = &event.Key{Name: "4e_checksum"}
	key4fMode     = &event.Key{Name: "4f_mode"}

	key5aMin     = &event.Key{Name: "5a_min"}
	key5bMix     = &event.Key{Name: "5b_mix"}
	key5cPort    = &event.Key{Name: "5c_port"}
	key5dMinHops = &event.Key{Name: "5d_min_hops"}
	key5eMaxHops = &event.Key{Name: "5e_max_hops"}
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
