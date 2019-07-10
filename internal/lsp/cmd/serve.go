// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"path/filepath"
	"strings"
	"time"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp"
	"golang.org/x/tools/internal/lsp/debug"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/lsp/telemetry/tag"
	"golang.org/x/tools/internal/lsp/telemetry/trace"
	"golang.org/x/tools/internal/tool"
)

// Serve is a struct that exposes the configurable parts of the LSP server as
// flags, in the right form for tool.Main to consume.
type Serve struct {
	Logfile string `flag:"logfile" help:"filename to log to. if value is \"auto\", then logging to a default output file is enabled"`
	Mode    string `flag:"mode" help:"no effect"`
	Port    int    `flag:"port" help:"port on which to run gopls for debugging purposes"`
	Address string `flag:"listen" help:"address on which to listen for remote connections"`
	Trace   bool   `flag:"rpc.trace" help:"Print the full rpc trace in lsp inspector format"`
	Debug   string `flag:"debug" help:"Serve debug information on the supplied address"`

	app *Application
}

func (s *Serve) Name() string  { return "serve" }
func (s *Serve) Usage() string { return "" }
func (s *Serve) ShortHelp() string {
	return "run a server for Go code using the Language Server Protocol"
}
func (s *Serve) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), `
The server communicates using JSONRPC2 on stdin and stdout, and is intended to be run directly as
a child of an editor process.

gopls server flags are:
`)
	f.PrintDefaults()
}

// Run configures a server based on the flags, and then runs it.
// It blocks until the server shuts down.
func (s *Serve) Run(ctx context.Context, args ...string) error {
	if len(args) > 0 {
		return tool.CommandLineErrorf("server does not take arguments, got %v", args)
	}
	out := os.Stderr
	if s.Logfile != "" {
		filename := s.Logfile
		if filename == "auto" {
			filename = filepath.Join(os.TempDir(), fmt.Sprintf("gopls-%d.log", os.Getpid()))
		}
		f, err := os.Create(filename)
		if err != nil {
			return fmt.Errorf("Unable to create log file: %v", err)
		}
		defer f.Close()
		log.SetOutput(io.MultiWriter(os.Stderr, f))
		out = f
	}

	debug.Serve(ctx, s.Debug)

	if s.app.Remote != "" {
		return s.forward()
	}

	// For debugging purposes only.
	run := func(ctx context.Context, srv *lsp.Server) {
		srv.Conn.AddHandler(&handler{loggingRPCs: s.Trace, out: out})
		go srv.Run(ctx)
	}
	if s.Address != "" {
		return lsp.RunServerOnAddress(ctx, s.app.cache, s.Address, run)
	}
	if s.Port != 0 {
		return lsp.RunServerOnPort(ctx, s.app.cache, s.Port, run)
	}
	stream := jsonrpc2.NewHeaderStream(os.Stdin, os.Stdout)
	ctx, srv := lsp.NewServer(ctx, s.app.cache, stream)
	srv.Conn.AddHandler(&handler{loggingRPCs: s.Trace, out: out})
	return srv.Run(ctx)
}

func (s *Serve) forward() error {
	conn, err := net.Dial("tcp", s.app.Remote)
	if err != nil {
		return err
	}
	errc := make(chan error)

	go func(conn net.Conn) {
		_, err := io.Copy(conn, os.Stdin)
		errc <- err
	}(conn)

	go func(conn net.Conn) {
		_, err := io.Copy(os.Stdout, conn)
		errc <- err
	}(conn)

	return <-errc
}

type handler struct {
	loggingRPCs bool
	out         io.Writer
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

func (h *handler) Deliver(ctx context.Context, r *jsonrpc2.Request, delivered bool) bool {
	stats := h.getStats(ctx)
	if stats != nil {
		stats.delivering()
	}
	return false
}

func (h *handler) Cancel(ctx context.Context, conn *jsonrpc2.Conn, id jsonrpc2.ID, cancelled bool) bool {
	return false
}

func (h *handler) Request(ctx context.Context, direction jsonrpc2.Direction, r *jsonrpc2.WireRequest) context.Context {
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
		tag.Tag{Key: telemetry.Method, Value: r.Method},
		tag.Tag{Key: telemetry.RPCDirection, Value: mode},
		tag.Tag{Key: telemetry.RPCID, Value: r.ID},
	)
	telemetry.Started.Record(ctx, 1)
	_, stats.delivering = trace.StartSpan(ctx, "queued")
	return ctx
}

func (h *handler) Response(ctx context.Context, direction jsonrpc2.Direction, r *jsonrpc2.WireResponse) context.Context {
	stats := h.getStats(ctx)
	h.logRPC(direction, r.ID, 0, stats.method, r.Result, nil)
	return ctx
}

func (h *handler) Done(ctx context.Context, err error) {
	stats := h.getStats(ctx)
	h.logRPC(stats.direction, stats.id, time.Since(stats.start), stats.method, stats.payload, err)
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

func (h *handler) Read(ctx context.Context, bytes int64) context.Context {
	telemetry.SentBytes.Record(ctx, bytes)
	return ctx
}

func (h *handler) Wrote(ctx context.Context, bytes int64) context.Context {
	telemetry.ReceivedBytes.Record(ctx, bytes)
	return ctx
}

const eol = "\r\n\r\n\r\n"

func (h *handler) Error(ctx context.Context, err error) {
	stats := h.getStats(ctx)
	h.logRPC(stats.direction, stats.id, 0, stats.method, nil, err)
}

func (h *handler) getStats(ctx context.Context) *rpcStats {
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

func (h *handler) logRPC(direction jsonrpc2.Direction, id *jsonrpc2.ID, elapsed time.Duration, method string, payload *json.RawMessage, err error) {
	if !h.loggingRPCs {
		return
	}
	const eol = "\r\n\r\n\r\n"
	if err != nil {
		fmt.Fprintf(h.out, "[Error - %v] %s %s%s %v%s", time.Now().Format("3:04:05 PM"),
			direction, method, id, err, eol)
		return
	}
	outx := new(strings.Builder)
	fmt.Fprintf(outx, "[Trace - %v] ", time.Now().Format("3:04:05 PM"))
	switch direction {
	case jsonrpc2.Send:
		fmt.Fprint(outx, "Received ")
	case jsonrpc2.Receive:
		fmt.Fprint(outx, "Sending ")
	}
	switch {
	case id == nil:
		fmt.Fprint(outx, "notification ")
	case elapsed >= 0:
		fmt.Fprint(outx, "response ")
	default:
		fmt.Fprint(outx, "request ")
	}
	fmt.Fprintf(outx, "'%s", method)
	switch {
	case id == nil:
		// do nothing
	case id.Name != "":
		fmt.Fprintf(outx, " - (%s)", id.Name)
	default:
		fmt.Fprintf(outx, " - (%d)", id.Number)
	}
	fmt.Fprint(outx, "'")
	if elapsed >= 0 {
		msec := int(elapsed.Round(time.Millisecond) / time.Millisecond)
		fmt.Fprintf(outx, " in %dms", msec)
	}
	params := "null"
	if payload != nil {
		params = string(*payload)
	}
	if params == "null" {
		params = "{}"
	}
	fmt.Fprintf(outx, ".\r\nParams: %s%s", params, eol)
	fmt.Fprintf(h.out, "%s", outx.String())
}
