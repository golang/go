// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package traceviewer

import (
	"fmt"
	"html/template"
	"math"
	"strings"
	"time"
)

// TimeHistogram is an high-dynamic-range histogram for durations.
type TimeHistogram struct {
	Count                int
	Buckets              []int
	MinBucket, MaxBucket int
}

// Five buckets for every power of 10.
var logDiv = math.Log(math.Pow(10, 1.0/5))

// Add adds a single sample to the histogram.
func (h *TimeHistogram) Add(d time.Duration) {
	var bucket int
	if d > 0 {
		bucket = int(math.Log(float64(d)) / logDiv)
	}
	if len(h.Buckets) <= bucket {
		h.Buckets = append(h.Buckets, make([]int, bucket-len(h.Buckets)+1)...)
		h.Buckets = h.Buckets[:cap(h.Buckets)]
	}
	h.Buckets[bucket]++
	if bucket < h.MinBucket || h.MaxBucket == 0 {
		h.MinBucket = bucket
	}
	if bucket > h.MaxBucket {
		h.MaxBucket = bucket
	}
	h.Count++
}

// BucketMin returns the minimum duration value for a provided bucket.
func (h *TimeHistogram) BucketMin(bucket int) time.Duration {
	return time.Duration(math.Exp(float64(bucket) * logDiv))
}

// ToHTML renders the histogram as HTML.
func (h *TimeHistogram) ToHTML(urlmaker func(min, max time.Duration) string) template.HTML {
	if h == nil || h.Count == 0 {
		return template.HTML("")
	}

	const barWidth = 400

	maxCount := 0
	for _, count := range h.Buckets {
		if count > maxCount {
			maxCount = count
		}
	}

	w := new(strings.Builder)
	fmt.Fprintf(w, `<table>`)
	for i := h.MinBucket; i <= h.MaxBucket; i++ {
		// Tick label.
		if h.Buckets[i] > 0 {
			fmt.Fprintf(w, `<tr><td class="histoTime" align="right"><a href=%s>%s</a></td>`, urlmaker(h.BucketMin(i), h.BucketMin(i+1)), h.BucketMin(i))
		} else {
			fmt.Fprintf(w, `<tr><td class="histoTime" align="right">%s</td>`, h.BucketMin(i))
		}
		// Bucket bar.
		width := h.Buckets[i] * barWidth / maxCount
		fmt.Fprintf(w, `<td><div style="width:%dpx;background:blue;position:relative">&nbsp;</div></td>`, width)
		// Bucket count.
		fmt.Fprintf(w, `<td align="right"><div style="position:relative">%d</div></td>`, h.Buckets[i])
		fmt.Fprintf(w, "</tr>\n")

	}
	// Final tick label.
	fmt.Fprintf(w, `<tr><td align="right">%s</td></tr>`, h.BucketMin(h.MaxBucket+1))
	fmt.Fprintf(w, `</table>`)
	return template.HTML(w.String())
}
