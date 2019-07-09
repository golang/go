// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package debug

import (
	"fmt"
	"html/template"
	"log"
	"net/http"
	"sort"

	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/lsp/telemetry/metric"
)

var rpcTmpl = template.Must(template.Must(BaseTemplate.Clone()).Parse(`
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
		<i>By bucket</i> 0s {{range .Latency.Values}}<b>{{.Count}}</b> {{.Limit}} {{end}}
		<br>
		<i>Received</i> {{with .Received}}{{.Mean}} ({{.Min}}<{{.Max}}){{end}}
		<i>Sent</i> {{with .Sent}}{{.Mean}} ({{.Min}}<{{.Max}}){{end}}
		<br>
		<i>Result codes</i> {{range .Codes}}{{.Key}}={{.Count}} {{end}}
		</P>
	{{end}}
{{end}}
`))

type rpcs struct {
	Inbound  []*rpcStats
	Outbound []*rpcStats
}

type rpcStats struct {
	Method     string
	Started    int64
	Completed  int64
	InProgress int64
	Latency    rpcTimeHistogram
	Received   rpcBytesHistogram
	Sent       rpcBytesHistogram
	Codes      []*rpcCodeBucket
}

type rpcTimeHistogram struct {
	Sum    timeUnits
	Count  int64
	Mean   timeUnits
	Min    timeUnits
	Max    timeUnits
	Values []rpcTimeBucket
}

type rpcTimeBucket struct {
	Limit timeUnits
	Count int64
}

type rpcBytesHistogram struct {
	Sum    byteUnits
	Count  int64
	Mean   byteUnits
	Min    byteUnits
	Max    byteUnits
	Values []rpcBytesBucket
}

type rpcBytesBucket struct {
	Limit byteUnits
	Count int64
}

type rpcCodeBucket struct {
	Key   string
	Count int64
}

func (r *rpcs) observeMetric(data metric.Data) {
	for i, group := range data.Groups() {
		set := &r.Inbound
		if group.Get(telemetry.RPCDirection) == telemetry.Outbound {
			set = &r.Outbound
		}
		method, ok := group.Get(telemetry.Method).(string)
		if !ok {
			log.Printf("Not a method... %v", group)
			continue
		}
		index := sort.Search(len(*set), func(i int) bool {
			return (*set)[i].Method >= method
		})
		if index >= len(*set) || (*set)[index].Method != method {
			old := *set
			*set = make([]*rpcStats, len(old)+1)
			copy(*set, old[:index])
			copy((*set)[index+1:], old[index:])
			(*set)[index] = &rpcStats{Method: method}
		}
		stats := (*set)[index]
		switch data.Handle() {
		case started:
			stats.Started = data.(*metric.Int64Data).Rows[i]
		case completed:
			status, ok := group.Get(telemetry.StatusCode).(string)
			if !ok {
				log.Printf("Not status... %v", group)
				continue
			}
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
					return stats.Codes[i].Key < stats.Codes[i].Key
				})
			}
			b.Count = data.(*metric.Int64Data).Rows[i]
		case latency:
			data := data.(*metric.HistogramFloat64Data)
			row := data.Rows[i]
			stats.Latency.Count = row.Count
			stats.Latency.Sum = timeUnits(row.Sum)
			stats.Latency.Min = timeUnits(row.Min)
			stats.Latency.Max = timeUnits(row.Max)
			stats.Latency.Mean = timeUnits(row.Sum) / timeUnits(row.Count)
			stats.Latency.Values = make([]rpcTimeBucket, len(data.Info.Buckets))
			last := int64(0)
			for i, b := range data.Info.Buckets {
				stats.Latency.Values[i].Limit = timeUnits(b)
				stats.Latency.Values[i].Count = row.Values[i] - last
				last = row.Values[i]
			}
		case sentBytes:
			data := data.(*metric.HistogramInt64Data)
			row := data.Rows[i]
			stats.Sent.Count = row.Count
			stats.Sent.Sum = byteUnits(row.Sum)
			stats.Sent.Min = byteUnits(row.Min)
			stats.Sent.Max = byteUnits(row.Max)
			stats.Sent.Mean = byteUnits(row.Sum) / byteUnits(row.Count)
		case receivedBytes:
			data := data.(*metric.HistogramInt64Data)
			row := data.Rows[i]
			stats.Received.Count = row.Count
			stats.Received.Sum = byteUnits(row.Sum)
			stats.Sent.Min = byteUnits(row.Min)
			stats.Sent.Max = byteUnits(row.Max)
			stats.Received.Mean = byteUnits(row.Sum) / byteUnits(row.Count)
		}
	}

	for _, set := range [][]*rpcStats{r.Inbound, r.Outbound} {
		for _, stats := range set {
			stats.Completed = 0
			for _, b := range stats.Codes {
				stats.Completed += b.Count
			}
			stats.InProgress = stats.Started - stats.Completed
		}
	}
}

func (r *rpcs) getData(req *http.Request) interface{} {
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
