// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package telemetry provides the hooks and adapters to allow use of telemetry
// throughout gopls.
package telemetry

import (
	"net/http"

	"golang.org/x/tools/internal/lsp/telemetry/stats"
	"golang.org/x/tools/internal/lsp/telemetry/tag"
)

const (
	// create the tag keys we use
	Method        = tag.Key("method")
	StatusCode    = tag.Key("status.code")
	StatusMessage = tag.Key("status.message")
	RPCID         = tag.Key("id")
	RPCDirection  = tag.Key("direction")
	File          = tag.Key("file")
	Package       = tag.Key("package")
)

var (
	Handle = func(mux *http.ServeMux) {}

	Started       = stats.NullInt64Measure()
	ReceivedBytes = stats.NullInt64Measure()
	SentBytes     = stats.NullInt64Measure()
	Latency       = stats.NullFloat64Measure()
)

const (
	Inbound  = "in"
	Outbound = "out"
)
