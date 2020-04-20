// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tag provides the labels used for telemetry throughout gopls.
package tag

import (
	"golang.org/x/tools/internal/event/core"
)

var (
	// create the label keys we use
	Method        = core.NewStringKey("method", "")
	StatusCode    = core.NewStringKey("status.code", "")
	StatusMessage = core.NewStringKey("status.message", "")
	RPCID         = core.NewStringKey("id", "")
	RPCDirection  = core.NewStringKey("direction", "")
	File          = core.NewStringKey("file", "")
	Directory     = core.NewKey("directory", "")
	URI           = core.NewKey("URI", "")
	Package       = core.NewStringKey("package", "")
	PackagePath   = core.NewStringKey("package_path", "")
	Query         = core.NewKey("query", "")
	Snapshot      = core.NewUInt64Key("snapshot", "")
	Operation     = core.NewStringKey("operation", "")

	Position     = core.NewKey("position", "")
	Category     = core.NewStringKey("category", "")
	PackageCount = core.NewIntKey("packages", "")
	Files        = core.NewKey("files", "")
	Port         = core.NewIntKey("port", "")
	Type         = core.NewKey("type", "")
	HoverKind    = core.NewStringKey("hoverkind", "")
)

var (
	// create the stats we measure
	Started       = core.NewInt64Key("started", "Count of started RPCs.")
	ReceivedBytes = core.NewInt64Key("received_bytes", "Bytes received.")            //, unit.Bytes)
	SentBytes     = core.NewInt64Key("sent_bytes", "Bytes sent.")                    //, unit.Bytes)
	Latency       = core.NewFloat64Key("latency_ms", "Elapsed time in milliseconds") //, unit.Milliseconds)
)

const (
	Inbound  = "in"
	Outbound = "out"
)
