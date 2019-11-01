// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package telemetry provides the hooks and adapters to allow use of telemetry
// throughout gopls.
package telemetry

import (
	"golang.org/x/tools/internal/telemetry/stats"
	"golang.org/x/tools/internal/telemetry/tag"
	"golang.org/x/tools/internal/telemetry/unit"
)

const (
	// create the tag keys we use
	Method        = tag.Key("method")
	StatusCode    = tag.Key("status.code")
	StatusMessage = tag.Key("status.message")
	RPCID         = tag.Key("id")
	RPCDirection  = tag.Key("direction")
	File          = tag.Key("file")
	Directory     = tag.Key("directory")
	URI           = tag.Key("URI")
	Package       = tag.Key("package")
	PackagePath   = tag.Key("package_path")
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
