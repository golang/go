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
	"time"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/debug"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/telemetry/trace"
	"golang.org/x/tools/internal/tool"
	errors "golang.org/x/xerrors"
)

// Serve is a struct that exposes the configurable parts of the LSP server as
// flags, in the right form for tool.Main to consume.
type Serve struct {
	Logfile string `flag:"logfile" help:"filename to log to. if value is \"auto\", then logging to a default output file is enabled"`
	Mode    string `flag:"mode" help:"no effect"`
	Port    int    `flag:"port" help:"port on which to run gopls for debugging purposes"`
	Address string `flag:"listen" help:"address on which to listen for remote connections"`
	Trace   bool   `flag:"rpc.trace" help:"print the full rpc trace in lsp inspector format"`
	Debug   string `flag:"debug" help:"serve debug information on the supplied address"`

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
	logfile := s.Logfile
	if logfile != "" {
		if logfile == "auto" {
			logfile = filepath.Join(os.TempDir(), fmt.Sprintf("gopls-%d.log", os.Getpid()))
		}
		f, err := os.Create(logfile)
		if err != nil {
			return errors.Errorf("Unable to create log file: %v", err)
		}
		defer f.Close()
		log.SetOutput(io.MultiWriter(os.Stderr, f))
		out = f
	}

	debug.Serve(ctx, s.Debug, debugServe{s: s, logfile: logfile, start: time.Now()})

	if s.app.Remote != "" {
		return s.forward()
	}

	prepare := func(ctx context.Context, srv *lsp.Server) *lsp.Server {
		srv.Conn.AddHandler(&handler{})
		return srv
	}
	run := func(ctx context.Context, srv *lsp.Server) { go prepare(ctx, srv).Run(ctx) }
	if s.Address != "" {
		return lsp.RunServerOnAddress(ctx, cache.New(s.app.options), s.Address, run)
	}
	if s.Port != 0 {
		return lsp.RunServerOnPort(ctx, cache.New(s.app.options), s.Port, run)
	}
	stream := jsonrpc2.NewHeaderStream(os.Stdin, os.Stdout)
	if s.Trace {
		stream = protocol.LoggingStream(stream, out)
	}
	ctx, srv := lsp.NewServer(ctx, cache.New(s.app.options).NewSession(), stream)
	return prepare(ctx, srv).Run(ctx)
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

// debugServe implements the debug.Instance interface.
type debugServe struct {
	s       *Serve
	logfile string
	start   time.Time
}

func (d debugServe) Logfile() string      { return d.logfile }
func (d debugServe) StartTime() time.Time { return d.start }
func (d debugServe) Port() int            { return d.s.Port }
func (d debugServe) Address() string      { return d.s.Address }
func (d debugServe) Debug() string        { return d.s.Debug }
func (d debugServe) Workdir() string      { return d.s.app.wd }

type handler struct{}

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

func (h *handler) Request(ctx context.Context, conn *jsonrpc2.Conn, direction jsonrpc2.Direction, r *jsonrpc2.WireRequest) context.Context {
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

func (h *handler) Response(ctx context.Context, conn *jsonrpc2.Conn, direction jsonrpc2.Direction, r *jsonrpc2.WireResponse) context.Context {
	return ctx
}

func (h *handler) Done(ctx context.Context, err error) {
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
