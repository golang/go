// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tag provides the labels used for telemetry throughout gopls.
package tag

import (
	"golang.org/x/tools/internal/telemetry/event"
	"golang.org/x/tools/internal/telemetry/stats"
	"golang.org/x/tools/internal/telemetry/unit"
)

var (
	// create the tag keys we use
	Method        = event.NewStringKey("method", "")
	StatusCode    = event.NewStringKey("status.code", "")
	StatusMessage = event.NewStringKey("status.message", "")
	RPCID         = event.NewStringKey("id", "")
	RPCDirection  = event.NewStringKey("direction", "")
	File          = event.NewStringKey("file", "")
	Directory     = event.NewKey("directory", "")
	URI           = event.NewKey("URI", "")
	Package       = event.NewStringKey("package", "")
	PackagePath   = event.NewStringKey("package_path", "")
	Query         = event.NewKey("query", "")
	Snapshot      = event.NewUInt64Key("snapshot", "")
	Operation     = event.NewStringKey("operation", "")

	Position     = event.NewKey("position", "")
	Category     = event.NewStringKey("category", "")
	PackageCount = event.NewIntKey("packages", "")
	Files        = event.NewKey("files", "")
	Port         = event.NewIntKey("port", "")
	Type         = event.NewKey("type", "")
	HoverKind    = event.NewStringKey("hoverkind", "")
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
