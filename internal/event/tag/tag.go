// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tag provides the labels used for telemetry throughout gopls.
package tag

import (
	"golang.org/x/tools/internal/event/keys"
)

var (
	// create the label keys we use
	Method        = keys.NewString("method", "")
	StatusCode    = keys.NewString("status.code", "")
	StatusMessage = keys.NewString("status.message", "")
	RPCID         = keys.NewString("id", "")
	RPCDirection  = keys.NewString("direction", "")
	File          = keys.NewString("file", "")
	Directory     = keys.New("directory", "")
	URI           = keys.New("URI", "")
	Package       = keys.NewString("package", "") // sorted comma-separated list of Package IDs
	PackagePath   = keys.NewString("package_path", "")
	Query         = keys.New("query", "")
	Snapshot      = keys.NewUInt64("snapshot", "")
	Operation     = keys.NewString("operation", "")

	Position     = keys.New("position", "")
	Category     = keys.NewString("category", "")
	PackageCount = keys.NewInt("packages", "")
	Files        = keys.New("files", "")
	Port         = keys.NewInt("port", "")
	Type         = keys.New("type", "")
	HoverKind    = keys.NewString("hoverkind", "")

	NewServer = keys.NewString("new_server", "A new server was added")
	EndServer = keys.NewString("end_server", "A server was shut down")

	ServerID     = keys.NewString("server", "The server ID an event is related to")
	Logfile      = keys.NewString("logfile", "")
	DebugAddress = keys.NewString("debug_address", "")
	GoplsPath    = keys.NewString("gopls_path", "")
	ClientID     = keys.NewString("client_id", "")

	Level = keys.NewInt("level", "The logging level")
)

var (
	// create the stats we measure
	Started       = keys.NewInt64("started", "Count of started RPCs.")
	ReceivedBytes = keys.NewInt64("received_bytes", "Bytes received.")            //, unit.Bytes)
	SentBytes     = keys.NewInt64("sent_bytes", "Bytes sent.")                    //, unit.Bytes)
	Latency       = keys.NewFloat64("latency_ms", "Elapsed time in milliseconds") //, unit.Milliseconds)
)

const (
	Inbound  = "in"
	Outbound = "out"
)
