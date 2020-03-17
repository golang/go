// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tag provides the labels used for telemetry throughout gopls.
package tag

import (
	"golang.org/x/tools/internal/telemetry/event"
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
	Started       = event.NewInt64Key("started", "Count of started RPCs.")
	ReceivedBytes = event.NewInt64Key("received_bytes", "Bytes received.")            //, unit.Bytes)
	SentBytes     = event.NewInt64Key("sent_bytes", "Bytes sent.")                    //, unit.Bytes)
	Latency       = event.NewFloat64Key("latency_ms", "Elapsed time in milliseconds") //, unit.Milliseconds)
)

const (
	Inbound  = "in"
	Outbound = "out"
)
