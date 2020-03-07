// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package export holds some exporter implementations.
// Larger more complex exporters are in sub packages of their own.
package export

import "golang.org/x/tools/internal/telemetry/event"

var (
	SetExporter = event.SetExporter
)
