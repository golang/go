// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package debug

import (
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/telemetry/metric"
)

var (
	// the distributions we use for histograms
	bytesDistribution        = []int64{1 << 10, 1 << 11, 1 << 12, 1 << 14, 1 << 16, 1 << 20}
	millisecondsDistribution = []float64{0.1, 0.5, 1, 2, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000}

	receivedBytes = metric.HistogramInt64{
		Name:        "received_bytes",
		Description: "Distribution of received bytes, by method.",
		Keys:        []interface{}{telemetry.RPCDirection, telemetry.Method},
		Buckets:     bytesDistribution,
	}.Record(telemetry.ReceivedBytes)

	sentBytes = metric.HistogramInt64{
		Name:        "sent_bytes",
		Description: "Distribution of sent bytes, by method.",
		Keys:        []interface{}{telemetry.RPCDirection, telemetry.Method},
		Buckets:     bytesDistribution,
	}.Record(telemetry.SentBytes)

	latency = metric.HistogramFloat64{
		Name:        "latency",
		Description: "Distribution of latency in milliseconds, by method.",
		Keys:        []interface{}{telemetry.RPCDirection, telemetry.Method},
		Buckets:     millisecondsDistribution,
	}.Record(telemetry.Latency)

	started = metric.Scalar{
		Name:        "started",
		Description: "Count of RPCs started by method.",
		Keys:        []interface{}{telemetry.RPCDirection, telemetry.Method},
	}.CountInt64(telemetry.Started)

	completed = metric.Scalar{
		Name:        "completed",
		Description: "Count of RPCs completed by method and status.",
		Keys:        []interface{}{telemetry.RPCDirection, telemetry.Method, telemetry.StatusCode},
	}.CountFloat64(telemetry.Latency)
)
