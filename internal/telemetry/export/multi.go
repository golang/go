// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package export

import (
	"context"

	"golang.org/x/tools/internal/telemetry"
)

// Multi returns an exporter that invokes all the exporters given to it in order.
func Multi(e ...Exporter) Exporter {
	a := make(multi, 0, len(e))
	for _, i := range e {
		if i == nil {
			continue
		}
		if i, ok := i.(multi); ok {
			a = append(a, i...)
			continue
		}
		a = append(a, i)
	}
	return a
}

type multi []Exporter

func (m multi) StartSpan(ctx context.Context, span *telemetry.Span) {
	for _, o := range m {
		o.StartSpan(ctx, span)
	}
}
func (m multi) FinishSpan(ctx context.Context, span *telemetry.Span) {
	for _, o := range m {
		o.FinishSpan(ctx, span)
	}
}
func (m multi) Log(ctx context.Context, event telemetry.Event) {
	for _, o := range m {
		o.Log(ctx, event)
	}
}
func (m multi) Metric(ctx context.Context, data telemetry.MetricData) {
	for _, o := range m {
		o.Metric(ctx, data)
	}
}
func (m multi) Flush() {
	for _, o := range m {
		o.Flush()
	}
}
