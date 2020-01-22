// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsprpc

import (
	"context"
	"encoding/json"
	"time"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/telemetry/trace"
)

type telemetryHandler struct{}

func (h telemetryHandler) Deliver(ctx context.Context, r *jsonrpc2.Request, delivered bool) bool {
	stats := h.getStats(ctx)
	if stats != nil {
		stats.delivering()
	}
	return false
}

func (h telemetryHandler) Cancel(ctx context.Context, conn *jsonrpc2.Conn, id jsonrpc2.ID, cancelled bool) bool {
	return false
}

func (h telemetryHandler) Request(ctx context.Context, conn *jsonrpc2.Conn, direction jsonrpc2.Direction, r *jsonrpc2.WireRequest) context.Context {
	if r.Method == "" {
		panic("no method in rpc stats")
	}
	stats := &rpcStats{
		method:    r.Method,
		start:     time.Now(),
		direction: direction,
		payload:   r.Params,
	}
	ctx = context.WithValue(ctx, statsKey, stats)
	mode := telemetry.Outbound
	if direction == jsonrpc2.Receive {
		mode = telemetry.Inbound
	}
	ctx, stats.close = trace.StartSpan(ctx, r.Method,
		telemetry.Method.Of(r.Method),
		telemetry.RPCDirection.Of(mode),
		telemetry.RPCID.Of(r.ID),
	)
	telemetry.Started.Record(ctx, 1)
	_, stats.delivering = trace.StartSpan(ctx, "queued")
	return ctx
}

func (h telemetryHandler) Response(ctx context.Context, conn *jsonrpc2.Conn, direction jsonrpc2.Direction, r *jsonrpc2.WireResponse) context.Context {
	return ctx
}

func (h telemetryHandler) Done(ctx context.Context, err error) {
	stats := h.getStats(ctx)
	if err != nil {
		ctx = telemetry.StatusCode.With(ctx, "ERROR")
	} else {
		ctx = telemetry.StatusCode.With(ctx, "OK")
	}
	elapsedTime := time.Since(stats.start)
	latencyMillis := float64(elapsedTime) / float64(time.Millisecond)
	telemetry.Latency.Record(ctx, latencyMillis)
	stats.close()
}

func (h telemetryHandler) Read(ctx context.Context, bytes int64) context.Context {
	telemetry.SentBytes.Record(ctx, bytes)
	return ctx
}

func (h telemetryHandler) Wrote(ctx context.Context, bytes int64) context.Context {
	telemetry.ReceivedBytes.Record(ctx, bytes)
	return ctx
}

const eol = "\r\n\r\n\r\n"

func (h telemetryHandler) Error(ctx context.Context, err error) {
}

func (h telemetryHandler) getStats(ctx context.Context) *rpcStats {
	stats, ok := ctx.Value(statsKey).(*rpcStats)
	if !ok || stats == nil {
		method, ok := ctx.Value(telemetry.Method).(string)
		if !ok {
			method = "???"
		}
		stats = &rpcStats{
			method: method,
			close:  func() {},
		}
	}
	return stats
}

type rpcStats struct {
	method     string
	direction  jsonrpc2.Direction
	id         *jsonrpc2.ID
	payload    *json.RawMessage
	start      time.Time
	delivering func()
	close      func()
}

type statsKeyType int

const statsKey = statsKeyType(0)
