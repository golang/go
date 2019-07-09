// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package debug

import (
	"bytes"
	"fmt"
	"net/http"
	"sort"

	"golang.org/x/tools/internal/lsp/telemetry/metric"
	"golang.org/x/tools/internal/lsp/telemetry/tag"
	"golang.org/x/tools/internal/lsp/telemetry/worker"
)

type prometheus struct {
	metrics []metric.Data
}

func (p *prometheus) observeMetric(data metric.Data) {
	name := data.Handle().Name()
	index := sort.Search(len(p.metrics), func(i int) bool {
		return p.metrics[i].Handle().Name() >= name
	})
	if index >= len(p.metrics) || p.metrics[index].Handle().Name() != name {
		old := p.metrics
		p.metrics = make([]metric.Data, len(old)+1)
		copy(p.metrics, old[:index])
		copy(p.metrics[index+1:], old[index:])
	}
	p.metrics[index] = data
}

func (p *prometheus) header(w http.ResponseWriter, name, description string, isGauge, isHistogram bool) {
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

func (p *prometheus) row(w http.ResponseWriter, name string, group tag.List, extra string, value interface{}) {
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

func (p *prometheus) serve(w http.ResponseWriter, r *http.Request) {
	done := make(chan struct{})
	worker.Do(func() {
		defer close(done)
		for _, data := range p.metrics {
			switch data := data.(type) {
			case *metric.Int64Data:
				p.header(w, data.Info.Name, data.Info.Description, data.IsGauge, false)
				for i, group := range data.Groups() {
					p.row(w, data.Info.Name, group, "", data.Rows[i])
				}

			case *metric.Float64Data:
				p.header(w, data.Info.Name, data.Info.Description, data.IsGauge, false)
				for i, group := range data.Groups() {
					p.row(w, data.Info.Name, group, "", data.Rows[i])
				}

			case *metric.HistogramInt64Data:
				p.header(w, data.Info.Name, data.Info.Description, false, true)
				for i, group := range data.Groups() {
					row := data.Rows[i]
					for j, b := range data.Info.Buckets {
						p.row(w, data.Info.Name+"_bucket", group, fmt.Sprintf(`le="%v"`, b), row.Values[j])
					}
					p.row(w, data.Info.Name+"_bucket", group, `le="+Inf"`, row.Count)
					p.row(w, data.Info.Name+"_count", group, "", row.Count)
					p.row(w, data.Info.Name+"_sum", group, "", row.Sum)
				}

			case *metric.HistogramFloat64Data:
				p.header(w, data.Info.Name, data.Info.Description, false, true)
				for i, group := range data.Groups() {
					row := data.Rows[i]
					for j, b := range data.Info.Buckets {
						p.row(w, data.Info.Name+"_bucket", group, fmt.Sprintf(`le="%v"`, b), row.Values[j])
					}
					p.row(w, data.Info.Name+"_bucket", group, `le="+Inf"`, row.Count)
					p.row(w, data.Info.Name+"_count", group, "", row.Count)
					p.row(w, data.Info.Name+"_sum", group, "", row.Sum)
				}
			}
		}
	})
	<-done
}
