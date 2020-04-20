// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package eventtest supports logging events to a test.
// You can use NewContext to create a context that knows how to deliver
// telemetry events back to the test.
// You must use this context or a derived one anywhere you want telemetry to be
// correctly routed back to the test it was constructed with.
// Any events delivered to a background context will be dropped.
//
// Importing this package will cause it to register a new global telemetry
// exporter that understands the special contexts returned by NewContext.
// This means you should not import this package if you are not going to call
// NewContext.
package eventtest

import (
	"bytes"
	"context"
	"sync"
	"testing"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/core"
	"golang.org/x/tools/internal/event/export"
	"golang.org/x/tools/internal/event/label"
)

func init() {
	e := &testExporter{buffer: &bytes.Buffer{}}
	e.logger = export.LogWriter(e.buffer, false)

	event.SetExporter(export.Spans(e.processEvent))
}

type testingKeyType int

const testingKey = testingKeyType(0)

// NewContext returns a context you should use for the active test.
func NewContext(ctx context.Context, t testing.TB) context.Context {
	return context.WithValue(ctx, testingKey, t)
}

type testExporter struct {
	mu     sync.Mutex
	buffer *bytes.Buffer
	logger event.Exporter
}

func (w *testExporter) processEvent(ctx context.Context, ev core.Event, tm label.Map) context.Context {
	w.mu.Lock()
	defer w.mu.Unlock()
	// build our log message in buffer
	result := w.logger(ctx, ev, tm)
	v := ctx.Value(testingKey)
	// get the testing.TB
	if w.buffer.Len() > 0 && v != nil {
		v.(testing.TB).Log(w.buffer)
	}
	w.buffer.Truncate(0)
	return result
}
