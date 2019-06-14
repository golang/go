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

var (
	Handle = func(mux *http.ServeMux) {}

	Started       = stats.NullInt64Measure()
	ReceivedBytes = stats.NullInt64Measure()
	SentBytes     = stats.NullInt64Measure()
	Latency       = stats.NullFloat64Measure()

	KeyRPCID        tag.Key
	KeyMethod       tag.Key
	KeyStatus       tag.Key
	KeyRPCDirection tag.Key
)

const (
	Inbound  = "in"
	Outbound = "out"
)
