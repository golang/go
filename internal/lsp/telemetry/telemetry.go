// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package telemetry provides the hooks and adapters to allow use of telemetry
// throughout gopls.
package telemetry

import (
	"golang.org/x/tools/internal/telemetry/event"
	"golang.org/x/tools/internal/telemetry/stats"
	"golang.org/x/tools/internal/telemetry/unit"
)

const (
	// create the tag keys we use
	Method        = event.Key("method")
	StatusCode    = event.Key("status.code")
	StatusMessage = event.Key("status.message")
	RPCID         = event.Key("id")
	RPCDirection  = event.Key("direction")
	File          = event.Key("file")
	Directory     = event.Key("directory")
	URI           = event.Key("URI")
	Package       = event.Key("package")
	PackagePath   = event.Key("package_path")
	Query         = event.Key("query")
	Snapshot      = event.Key("snapshot")
)

var (
	// create the stats we measure
	Started       = stats.Int64("started", "Count of started RPCs.", unit.Dimensionless)
	ReceivedBytes = stats.Int64("received_bytes", "Bytes received.", unit.Bytes)
	SentBytes     = stats.Int64("sent_bytes", "Bytes sent.", unit.Bytes)
	Latency       = stats.Float64("latency_ms", "Elapsed time in milliseconds", unit.Milliseconds)
)

const (
	Inbound  = "in"
	Outbound = "out"
)
