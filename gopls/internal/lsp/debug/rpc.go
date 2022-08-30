// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package debug

import (
	"context"
	"fmt"
	"html/template"
	"net/http"
	"sort"
	"sync"
	"time"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/core"
	"golang.org/x/tools/internal/event/export"
	"golang.org/x/tools/internal/event/label"
	"golang.org/x/tools/internal/event/tag"
)

var RPCTmpl = template.Must(template.Must(BaseTemplate.Clone()).Parse(`
{{define "title"}}RPC Information{{end}}
{{define "body"}}
	<H2>Inbound</H2>
	{{template "rpcSection" .Inbound}}
	<H2>Outbound</H2>
	{{template "rpcSection" .Outbound}}
{{end}}
{{define "rpcSection"}}
	{{range .}}<P>
		<b>{{.Method}}</b> {{.Started}} <a href="/trace/{{.Method}}">traces</a> ({{.InProgress}} in progress)
		<br>
		<i>Latency</i> {{with .Latency}}{{.Mean}} ({{.Min}}<{{.Max}}){{end}}
		<i>By bucket</i> 0s {{range .Latency.Values}}{{if gt .Count 0}}<b>{{.Count}}</b> {{.Limit}} {{end}}{{end}}
		<br>
		<i>Received</i> {{.Received}} (avg. {{.ReceivedMean}})
		<i>Sent</i> {{.Sent}} (avg. {{.SentMean}})
		<br>
		<i>Result codes</i> {{range .Codes}}{{.Key}}={{.Count}} {{end}}
		</P>
	{{end}}
{{end}}
`))

type Rpcs struct { // exported for testing
	mu       sync.Mutex
	Inbound  []*rpcStats // stats for incoming lsp rpcs sorted by method name
	Outbound []*rpcStats // stats for outgoing lsp rpcs sorted by method name
}

type rpcStats struct {
	Method    string
	Started   int64
	Completed int64

	Latency  rpcTimeHistogram
	Received byteUnits
	Sent     byteUnits
	Codes    []*rpcCodeBucket
}

type rpcTimeHistogram struct {
	Sum    timeUnits
	Count  int64
	Min    timeUnits
	Max    timeUnits
	Values []rpcTimeBucket
}

type rpcTimeBucket struct {
	Limit timeUnits
	Count int64
}

type rpcCodeBucket struct {
	Key   string
	Count int64
}

func (r *Rpcs) ProcessEvent(ctx context.Context, ev core.Event, lm label.Map) context.Context {
	r.mu.Lock()
	defer r.mu.Unlock()
	switch {
	case event.IsStart(ev):
		if _, stats := r.getRPCSpan(ctx, ev); stats != nil {
			stats.Started++
		}
	case event.IsEnd(ev):
		span, stats := r.getRPCSpan(ctx, ev)
		if stats != nil {
			endRPC(ctx, ev, span, stats)
		}
	case event.IsMetric(ev):
		sent := byteUnits(tag.SentBytes.Get(lm))
		rec := byteUnits(tag.ReceivedBytes.Get(lm))
		if sent != 0 || rec != 0 {
			if _, stats := r.getRPCSpan(ctx, ev); stats != nil {
				stats.Sent += sent
				stats.Received += rec
			}
		}
	}
	return ctx
}

func endRPC(ctx context.Context, ev core.Event, span *export.Span, stats *rpcStats) {
	// update the basic counts
	stats.Completed++

	// get and record the status code
	if status := getStatusCode(span); status != "" {
		var b *rpcCodeBucket
		for c, entry := range stats.Codes {
			if entry.Key == status {
				b = stats.Codes[c]
				break
			}
		}
		if b == nil {
			b = &rpcCodeBucket{Key: status}
			stats.Codes = append(stats.Codes, b)
			sort.Slice(stats.Codes, func(i int, j int) bool {
				return stats.Codes[i].Key < stats.Codes[j].Key
			})
		}
		b.Count++
	}

	// calculate latency if this was an rpc span
	elapsedTime := span.Finish().At().Sub(span.Start().At())
	latencyMillis := timeUnits(elapsedTime) / timeUnits(time.Millisecond)
	if stats.Latency.Count == 0 {
		stats.Latency.Min = latencyMillis
		stats.Latency.Max = latencyMillis
	} else {
		if stats.Latency.Min > latencyMillis {
			stats.Latency.Min = latencyMillis
		}
		if stats.Latency.Max < latencyMillis {
			stats.Latency.Max = latencyMillis
		}
	}
	stats.Latency.Count++
	stats.Latency.Sum += latencyMillis
	for i := range stats.Latency.Values {
		if stats.Latency.Values[i].Limit > latencyMillis {
			stats.Latency.Values[i].Count++
			break
		}
	}
}

func (r *Rpcs) getRPCSpan(ctx context.Context, ev core.Event) (*export.Span, *rpcStats) {
	// get the span
	span := export.GetSpan(ctx)
	if span == nil {
		return nil, nil
	}
	// use the span start event look up the correct stats block
	// we do this because it prevents us matching a sub span
	return span, r.getRPCStats(span.Start())
}

func (r *Rpcs) getRPCStats(lm label.Map) *rpcStats {
	method := tag.Method.Get(lm)
	if method == "" {
		return nil
	}
	set := &r.Inbound
	if tag.RPCDirection.Get(lm) != tag.Inbound {
		set = &r.Outbound
	}
	// get the record for this method
	index := sort.Search(len(*set), func(i int) bool {
		return (*set)[i].Method >= method
	})

	if index < len(*set) && (*set)[index].Method == method {
		return (*set)[index]
	}

	old := *set
	*set = make([]*rpcStats, len(old)+1)
	copy(*set, old[:index])
	copy((*set)[index+1:], old[index:])
	stats := &rpcStats{Method: method}
	stats.Latency.Values = make([]rpcTimeBucket, len(millisecondsDistribution))
	for i, m := range millisecondsDistribution {
		stats.Latency.Values[i].Limit = timeUnits(m)
	}
	(*set)[index] = stats
	return stats
}

func (s *rpcStats) InProgress() int64       { return s.Started - s.Completed }
func (s *rpcStats) SentMean() byteUnits     { return s.Sent / byteUnits(s.Started) }
func (s *rpcStats) ReceivedMean() byteUnits { return s.Received / byteUnits(s.Started) }

func (h *rpcTimeHistogram) Mean() timeUnits { return h.Sum / timeUnits(h.Count) }

func getStatusCode(span *export.Span) string {
	for _, ev := range span.Events() {
		if status := tag.StatusCode.Get(ev); status != "" {
			return status
		}
	}
	return ""
}

func (r *Rpcs) getData(req *http.Request) interface{} {
	return r
}

func units(v float64, suffixes []string) string {
	s := ""
	for _, s = range suffixes {
		n := v / 1000
		if n < 1 {
			break
		}
		v = n
	}
	return fmt.Sprintf("%.2f%s", v, s)
}

type timeUnits float64

func (v timeUnits) String() string {
	v = v * 1000 * 1000
	return units(float64(v), []string{"ns", "μs", "ms", "s"})
}

type byteUnits float64

func (v byteUnits) String() string {
	return units(float64(v), []string{"B", "KB", "MB", "GB", "TB"})
}
