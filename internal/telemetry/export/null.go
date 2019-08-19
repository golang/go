// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package export

import (
	"context"

	"golang.org/x/tools/internal/telemetry"
)

// Null returns an observer that does nothing.
func Null() Exporter {
	return null{}
}

type null struct{}

func (null) StartSpan(context.Context, *telemetry.Span)   {}
func (null) FinishSpan(context.Context, *telemetry.Span)  {}
func (null) Log(context.Context, telemetry.Event)         {}
func (null) Metric(context.Context, telemetry.MetricData) {}
func (null) Flush()                                       {}
