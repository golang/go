// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package export holds the definition of the telemetry Exporter interface,
// along with some simple implementations.
// Larger more complex exporters are in sub packages of their own.
package export

import (
	"context"
	"os"
	"sync/atomic"
	"unsafe"

	"golang.org/x/tools/internal/telemetry"
)

type Exporter interface {
	// ProcessEvent is a function that handles all events.
	// Exporters may use information in the context to decide what to do with a
	// given event.
	ProcessEvent(context.Context, telemetry.Event) context.Context

	Metric(context.Context, telemetry.MetricData)
}

var (
	exporter unsafe.Pointer
)

func init() {
	SetExporter(LogWriter(os.Stderr, true))
}

func SetExporter(e Exporter) {
	p := unsafe.Pointer(&e)
	if e == nil {
		p = nil
	}
	atomic.StorePointer(&exporter, p)
}

func ProcessEvent(ctx context.Context, event telemetry.Event) context.Context {
	exporterPtr := (*Exporter)(atomic.LoadPointer(&exporter))
	if exporterPtr == nil {
		return ctx
	}
	// and now also hand the event of to the current exporter
	return (*exporterPtr).ProcessEvent(ctx, event)
}

func Metric(ctx context.Context, data telemetry.MetricData) {
	exporterPtr := (*Exporter)(atomic.LoadPointer(&exporter))
	if exporterPtr == nil {
		return
	}
	(*exporterPtr).Metric(ctx, data)
}
