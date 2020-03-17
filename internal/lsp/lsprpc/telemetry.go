// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsprpc

import (
	"context"
	"encoding/json"
	"time"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/telemetry/event"
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
		id:        r.ID,
		start:     time.Now(),
		direction: direction,
		payload:   r.Params,
	}
	ctx = context.WithValue(ctx, statsKey, stats)
	mode := tag.Outbound
	if direction == jsonrpc2.Receive {
		mode = tag.Inbound
	}
	ctx, stats.close = event.StartSpan(ctx, r.Method,
		tag.Method.Of(r.Method),
		tag.RPCDirection.Of(mode),
		tag.RPCID.Of(r.ID.String()),
	)
	event.Record(ctx, tag.Started.Of(1))
	_, stats.delivering = event.StartSpan(ctx, "queued")
	return ctx
}

func (h telemetryHandler) Response(ctx context.Context, conn *jsonrpc2.Conn, direction jsonrpc2.Direction, r *jsonrpc2.WireResponse) context.Context {
	return ctx
}

func (h telemetryHandler) Done(ctx context.Context, err error) {
	stats := h.getStats(ctx)
	if err != nil {
		ctx = event.Label(ctx, tag.StatusCode.Of("ERROR"))
	} else {
		ctx = event.Label(ctx, tag.StatusCode.Of("OK"))
	}
	elapsedTime := time.Since(stats.start)
	latencyMillis := float64(elapsedTime) / float64(time.Millisecond)
	event.Record(ctx, tag.Latency.Of(latencyMillis))
	stats.close()
}

func (h telemetryHandler) Read(ctx context.Context, bytes int64) context.Context {
	event.Record(ctx, tag.SentBytes.Of(bytes))
	return ctx
}

func (h telemetryHandler) Wrote(ctx context.Context, bytes int64) context.Context {
	event.Record(ctx, tag.ReceivedBytes.Of(bytes))
	return ctx
}

func (h telemetryHandler) Error(ctx context.Context, err error) {
}

func (h telemetryHandler) getStats(ctx context.Context) *rpcStats {
	stats, ok := ctx.Value(statsKey).(*rpcStats)
	if !ok || stats == nil {
		method, ok := ctx.Value(tag.Method).(string)
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
