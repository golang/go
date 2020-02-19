// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package lsprpc implements a jsonrpc2.StreamServer that may be used to
// serve the LSP on a jsonrpc2 channel.
package lsprpc

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"strconv"
	"sync/atomic"
	"time"

	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/debug"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/telemetry/log"
)

// The StreamServer type is a jsonrpc2.StreamServer that handles incoming
// streams as a new LSP session, using a shared cache.
type StreamServer struct {
	withTelemetry bool
	debug         *debug.Instance
	cache         *cache.Cache

	// serverForTest may be set to a test fake for testing.
	serverForTest protocol.Server
}

var clientIndex, serverIndex int64

// NewStreamServer creates a StreamServer using the shared cache. If
// withTelemetry is true, each session is instrumented with telemetry that
// records RPC statistics.
func NewStreamServer(cache *cache.Cache, withTelemetry bool, debugInstance *debug.Instance) *StreamServer {
	s := &StreamServer{
		withTelemetry: withTelemetry,
		debug:         debugInstance,
		cache:         cache,
	}
	return s
}

// debugInstance is the common functionality shared between client and server
// gopls instances.
type debugInstance struct {
	id           string
	debugAddress string
	logfile      string
}

func (d debugInstance) ID() string {
	return d.id
}

func (d debugInstance) DebugAddress() string {
	return d.debugAddress
}

func (d debugInstance) Logfile() string {
	return d.logfile
}

// A debugServer is held by the client to identity the remove server to which
// it is connected.
type debugServer struct {
	debugInstance
	// clientID is the id of this client on the server.
	clientID string
}

func (s debugServer) ClientID() string {
	return s.clientID
}

// A debugClient is held by the server to identify an incoming client
// connection.
type debugClient struct {
	debugInstance
	// session is the session serving this client.
	session *cache.Session
	// serverID is this id of this server on the client.
	serverID string
}

func (c debugClient) Session() debug.Session {
	return cache.DebugSession{Session: c.session}
}

func (c debugClient) ServerID() string {
	return c.serverID
}

// ServeStream implements the jsonrpc2.StreamServer interface, by handling
// incoming streams using a new lsp server.
func (s *StreamServer) ServeStream(ctx context.Context, stream jsonrpc2.Stream) error {
	index := atomic.AddInt64(&clientIndex, 1)

	conn := jsonrpc2.NewConn(stream)
	client := protocol.ClientDispatcher(conn)
	session := s.cache.NewSession()
	dc := &debugClient{
		debugInstance: debugInstance{
			id: strconv.FormatInt(index, 10),
		},
		session: session,
	}
	s.debug.State.AddClient(dc)
	server := s.serverForTest
	if server == nil {
		server = lsp.NewServer(session, client)
	}
	conn.AddHandler(protocol.ServerHandler(server))
	conn.AddHandler(protocol.Canceller{})
	if s.withTelemetry {
		conn.AddHandler(telemetryHandler{})
	}
	conn.AddHandler(&handshaker{
		client: dc,
		debug:  s.debug,
	})
	return conn.Run(protocol.WithClient(ctx, client))
}

// A Forwarder is a jsonrpc2.StreamServer that handles an LSP stream by
// forwarding it to a remote. This is used when the gopls process started by
// the editor is in the `-remote` mode, which means it finds and connects to a
// separate gopls daemon. In these cases, we still want the forwarder gopls to
// be instrumented with telemetry, and want to be able to in some cases hijack
// the jsonrpc2 connection with the daemon.
type Forwarder struct {
	network, addr string

	// Configuration. Right now, not all of this may be customizable, but in the
	// future it probably will be.
	withTelemetry bool
	dialTimeout   time.Duration
	retries       int
	debug         *debug.Instance
}

// NewForwarder creates a new Forwarder, ready to forward connections to the
// remote server specified by network and addr.
func NewForwarder(network, addr string, withTelemetry bool, debugInstance *debug.Instance) *Forwarder {
	return &Forwarder{
		network:       network,
		addr:          addr,
		withTelemetry: withTelemetry,
		dialTimeout:   1 * time.Second,
		retries:       5,
		debug:         debugInstance,
	}
}

// ServeStream dials the forwarder remote and binds the remote to serve the LSP
// on the incoming stream.
func (f *Forwarder) ServeStream(ctx context.Context, stream jsonrpc2.Stream) error {
	clientConn := jsonrpc2.NewConn(stream)
	client := protocol.ClientDispatcher(clientConn)

	var (
		netConn net.Conn
		err     error
	)
	// Sometimes the forwarder will be started immediately after the server is
	// started. To account for these cases, add in some simple retrying.
	// Note that the number of total attempts is f.retries + 1.
	for attempt := 0; attempt <= f.retries; attempt++ {
		startDial := time.Now()
		netConn, err = net.DialTimeout(f.network, f.addr, f.dialTimeout)
		if err == nil {
			break
		}
		log.Print(ctx, fmt.Sprintf("failed an attempt to connect to remote: %v\n", err))
		// In case our failure was a fast-failure, ensure we wait at least
		// f.dialTimeout before trying again.
		if attempt != f.retries {
			time.Sleep(f.dialTimeout - time.Since(startDial))
		}
	}
	if err != nil {
		return fmt.Errorf("forwarder: dialing remote: %v", err)
	}
	serverConn := jsonrpc2.NewConn(jsonrpc2.NewHeaderStream(netConn, netConn))
	server := protocol.ServerDispatcher(serverConn)

	// Forward between connections.
	serverConn.AddHandler(protocol.ClientHandler(client))
	serverConn.AddHandler(protocol.Canceller{})
	clientConn.AddHandler(protocol.ServerHandler(server))
	clientConn.AddHandler(protocol.Canceller{})
	clientConn.AddHandler(forwarderHandler{})
	if f.withTelemetry {
		clientConn.AddHandler(telemetryHandler{})
	}
	g, ctx := errgroup.WithContext(ctx)
	g.Go(func() error {
		return serverConn.Run(ctx)
	})
	g.Go(func() error {
		return clientConn.Run(ctx)
	})

	// Do a handshake with the server instance to exchange debug information.
	index := atomic.AddInt64(&serverIndex, 1)
	serverID := strconv.FormatInt(index, 10)
	var (
		hreq = handshakeRequest{
			ServerID:  serverID,
			Logfile:   f.debug.Logfile,
			DebugAddr: f.debug.DebugAddress,
		}
		hresp handshakeResponse
	)
	if err := serverConn.Call(ctx, handshakeMethod, hreq, &hresp); err != nil {
		log.Error(ctx, "gopls handshake failed", err)
	}
	f.debug.State.AddServer(debugServer{
		debugInstance: debugInstance{
			id:           serverID,
			logfile:      hresp.Logfile,
			debugAddress: hresp.DebugAddr,
		},
		clientID: hresp.ClientID,
	})
	return g.Wait()
}

// ForwarderExitFunc is used to exit the forwarder process. It is mutable for
// testing purposes.
var ForwarderExitFunc = os.Exit

// OverrideExitFuncsForTest can be used from test code to prevent the test
// process from exiting on server shutdown. The returned func reverts the exit
// funcs to their previous state.
func OverrideExitFuncsForTest() func() {
	// Override functions that would shut down the test process
	cleanup := func(lspExit, forwarderExit func(code int)) func() {
		return func() {
			lsp.ServerExitFunc = lspExit
			ForwarderExitFunc = forwarderExit
		}
	}(lsp.ServerExitFunc, ForwarderExitFunc)
	// It is an error for a test to shutdown a server process.
	lsp.ServerExitFunc = func(code int) {
		panic(fmt.Sprintf("LSP server exited with code %d", code))
	}
	// We don't want our forwarders to exit, but it's OK if they would have.
	ForwarderExitFunc = func(code int) {}
	return cleanup
}

// forwarderHandler intercepts 'exit' messages to prevent the shared gopls
// instance from exiting. In the future it may also intercept 'shutdown' to
// provide more graceful shutdown of the client connection.
type forwarderHandler struct {
	jsonrpc2.EmptyHandler
}

func (forwarderHandler) Deliver(ctx context.Context, r *jsonrpc2.Request, delivered bool) bool {
	// TODO(golang.org/issues/34111): we should more gracefully disconnect here,
	// once that process exists.
	if r.Method == "exit" {
		ForwarderExitFunc(0)
		// Still return true here to prevent the message from being delivered: in
		// tests, ForwarderExitFunc may be overridden to something that doesn't
		// exit the process.
		return true
	}
	return false
}

type handshaker struct {
	jsonrpc2.EmptyHandler
	client *debugClient
	debug  *debug.Instance
}

type handshakeRequest struct {
	ServerID  string `json:"serverID"`
	Logfile   string `json:"logfile"`
	DebugAddr string `json:"debugAddr"`
}

type handshakeResponse struct {
	ClientID  string `json:"clientID"`
	SessionID string `json:"sessionID"`
	Logfile   string `json:"logfile"`
	DebugAddr string `json:"debugAddr"`
}

const handshakeMethod = "gopls/handshake"

func (h *handshaker) Deliver(ctx context.Context, r *jsonrpc2.Request, delivered bool) bool {
	if r.Method == handshakeMethod {
		var req handshakeRequest
		if err := json.Unmarshal(*r.Params, &req); err != nil {
			sendError(ctx, r, err)
			return true
		}
		h.client.debugAddress = req.DebugAddr
		h.client.logfile = req.Logfile
		h.client.serverID = req.ServerID
		resp := handshakeResponse{
			ClientID:  h.client.id,
			SessionID: cache.DebugSession{Session: h.client.session}.ID(),
			Logfile:   h.debug.Logfile,
			DebugAddr: h.debug.DebugAddress,
		}
		if err := r.Reply(ctx, resp, nil); err != nil {
			log.Error(ctx, "replying to handshake", err)
		}
		return true
	}
	return false
}

func sendError(ctx context.Context, req *jsonrpc2.Request, err error) {
	if _, ok := err.(*jsonrpc2.Error); !ok {
		err = jsonrpc2.NewErrorf(jsonrpc2.CodeParseError, "%v", err)
	}
	if err := req.Reply(ctx, nil, err); err != nil {
		log.Error(ctx, "", err)
	}
}
