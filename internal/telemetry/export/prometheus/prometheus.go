// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package prometheus

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"sort"
	"sync"

	"golang.org/x/tools/internal/telemetry"
	"golang.org/x/tools/internal/telemetry/metric"
)

func New() *Exporter {
	return &Exporter{}
}

type Exporter struct {
	mu      sync.Mutex
	metrics []telemetry.MetricData
}

func (e *Exporter) StartSpan(ctx context.Context, span *telemetry.Span)  {}
func (e *Exporter) FinishSpan(ctx context.Context, span *telemetry.Span) {}
func (e *Exporter) Log(ctx context.Context, event telemetry.Event)       {}
func (e *Exporter) Flush()                                               {}

func (e *Exporter) Metric(ctx context.Context, data telemetry.MetricData) {
	e.mu.Lock()
	defer e.mu.Unlock()
	name := data.Handle()
	// We keep the metrics in name sorted order so the page is stable and easy
	// to read. We do this with an insertion sort rather than sorting the list
	// each time
	index := sort.Search(len(e.metrics), func(i int) bool {
		return e.metrics[i].Handle() >= name
	})
	if index >= len(e.metrics) || e.metrics[index].Handle() != name {
		// we have a new metric, so we need to make a space for it
		old := e.metrics
		e.metrics = make([]telemetry.MetricData, len(old)+1)
		copy(e.metrics, old[:index])
		copy(e.metrics[index+1:], old[index:])
	}
	e.metrics[index] = data
}

func (e *Exporter) header(w http.ResponseWriter, name, description string, isGauge, isHistogram bool) {
	kind := "counter"
	if isGauge {
		kind = "gauge"
	}
	if isHistogram {
		kind = "histogram"
	}
	fmt.Fprintf(w, "# HELP %s %s\n", name, description)
	fmt.Fprintf(w, "# TYPE %s %s\n", name, kind)
}

func (e *Exporter) row(w http.ResponseWriter, name string, group telemetry.TagList, extra string, value interface{}) {
	fmt.Fprint(w, name)
	buf := &bytes.Buffer{}
	fmt.Fprint(buf, group)
	if extra != "" {
		if buf.Len() > 0 {
			fmt.Fprint(buf, ",")
		}
		fmt.Fprint(buf, extra)
	}
	if buf.Len() > 0 {
		fmt.Fprint(w, "{")
		buf.WriteTo(w)
		fmt.Fprint(w, "}")
	}
	fmt.Fprintf(w, " %v\n", value)
}

func (e *Exporter) Serve(w http.ResponseWriter, r *http.Request) {
	e.mu.Lock()
	defer e.mu.Unlock()
	for _, data := range e.metrics {
		switch data := data.(type) {
		case *metric.Int64Data:
			e.header(w, data.Info.Name, data.Info.Description, data.IsGauge, false)
			for i, group := range data.Groups() {
				e.row(w, data.Info.Name, group, "", data.Rows[i])
			}

		case *metric.Float64Data:
			e.header(w, data.Info.Name, data.Info.Description, data.IsGauge, false)
			for i, group := range data.Groups() {
				e.row(w, data.Info.Name, group, "", data.Rows[i])
			}

		case *metric.HistogramInt64Data:
			e.header(w, data.Info.Name, data.Info.Description, false, true)
			for i, group := range data.Groups() {
				row := data.Rows[i]
				for j, b := range data.Info.Buckets {
					e.row(w, data.Info.Name+"_bucket", group, fmt.Sprintf(`le="%v"`, b), row.Values[j])
				}
				e.row(w, data.Info.Name+"_bucket", group, `le="+Inf"`, row.Count)
				e.row(w, data.Info.Name+"_count", group, "", row.Count)
				e.row(w, data.Info.Name+"_sum", group, "", row.Sum)
			}

		case *metric.HistogramFloat64Data:
			e.header(w, data.Info.Name, data.Info.Description, false, true)
			for i, group := range data.Groups() {
				row := data.Rows[i]
				for j, b := range data.Info.Buckets {
					e.row(w, data.Info.Name+"_bucket", group, fmt.Sprintf(`le="%v"`, b), row.Values[j])
				}
				e.row(w, data.Info.Name+"_bucket", group, `le="+Inf"`, row.Count)
				e.row(w, data.Info.Name+"_count", group, "", row.Count)
				e.row(w, data.Info.Name+"_sum", group, "", row.Sum)
			}
		}
	}
}
