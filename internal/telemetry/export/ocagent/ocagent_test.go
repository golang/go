// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ocagent_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"sync"
	"testing"
	"time"

	"golang.org/x/tools/internal/telemetry/export/ocagent"
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
	exporter *ocagent.Exporter
	sent     fakeSender
	start    time.Time
	at       time.Time
	end      time.Time
)

func init() {
	cfg := ocagent.Config{
		Host:    "tester",
		Process: 1,
		Service: "ocagent-tests",
		Client:  &http.Client{Transport: &sent},
	}
	cfg.Start, _ = time.Parse(time.RFC3339Nano, "1970-01-01T00:00:00Z")
	exporter = ocagent.Connect(&cfg)
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
