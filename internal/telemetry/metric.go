// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package telemetry

// Data represents a single point in the time series of a metric.
// This provides the common interface to all metrics no matter their data
// format.
// To get the actual values for the metric you must type assert to a concrete
// metric type.
type MetricData interface {
	// Handle returns the metric handle this data is for.
	Handle() string
	// Groups reports the rows that currently exist for this metric.
	Groups() []TagList
}
