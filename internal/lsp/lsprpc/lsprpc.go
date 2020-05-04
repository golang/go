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
	"log"
	"net"
	"os"
	"strconv"
	"sync/atomic"
	"time"

	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/debug"
	"golang.org/x/tools/internal/lsp/protocol"
)

// AutoNetwork is the pseudo network type used to signal that gopls should use
// automatic discovery to resolve a remote address.
const AutoNetwork = "auto"

// Unique identifiers for client/server.
var clientIndex, serverIndex int64

// The StreamServer type is a jsonrpc2.StreamServer that handles incoming
// streams as a new LSP session, using a shared cache.
type StreamServer struct {
	cache *cache.Cache

	// serverForTest may be set to a test fake for testing.
	serverForTest protocol.Server
}

// NewStreamServer creates a StreamServer using the shared cache. If
// withTelemetry is true, each session is instrumented with telemetry that
// records RPC statistics.
func NewStreamServer(cache *cache.Cache) *StreamServer {
	return &StreamServer{cache: cache}
}

// debugInstance is the common functionality shared between client and server
// gopls instances.
type debugInstance struct {
	id           string
	debugAddress string
	logfile      string
	goplsPath    string
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

func (d debugInstance) GoplsPath() string {
	return d.goplsPath
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
	session := s.cache.NewSession(ctx)
	dc := &debugClient{
		debugInstance: debugInstance{
			id: strconv.FormatInt(index, 10),
		},
		session: session,
	}
	if di := debug.GetInstance(ctx); di != nil {
		di.State.AddClient(dc)
		defer di.State.DropClient(dc)
	}
	server := s.serverForTest
	if server == nil {
		server = lsp.NewServer(session, client)
	}
	// Clients may or may not send a shutdown message. Make sure the server is
	// shut down.
	// TODO(rFindley): this shutdown should perhaps be on a disconnected context.
	defer func() {
		if err := server.Shutdown(ctx); err != nil {
			event.Error(ctx, "error shutting down", err)
		}
	}()
	executable, err := os.Executable()
	if err != nil {
		log.Printf("error getting gopls path: %v", err)
		executable = ""
	}
	ctx = protocol.WithClient(ctx, client)
	return conn.Run(ctx,
		protocol.Handlers(
			handshaker(dc, executable,
				protocol.ServerHandler(server,
					jsonrpc2.MethodNotFound))))
}

// A Forwarder is a jsonrpc2.StreamServer that handles an LSP stream by
// forwarding it to a remote. This is used when the gopls process started by
// the editor is in the `-remote` mode, which means it finds and connects to a
// separate gopls daemon. In these cases, we still want the forwarder gopls to
// be instrumented with telemetry, and want to be able to in some cases hijack
// the jsonrpc2 connection with the daemon.
type Forwarder struct {
	network, addr string

	// goplsPath is the path to the current executing gopls binary.
	goplsPath string

	// configuration
	dialTimeout         time.Duration
	retries             int
	remoteDebug         string
	remoteListenTimeout time.Duration
	remoteLogfile       string
}

// A ForwarderOption configures the behavior of the LSP forwarder.
type ForwarderOption interface {
	setForwarder(*Forwarder)
}

// RemoteDebugAddress configures the address used by the auto-started Gopls daemon
// for serving debug information.
type RemoteDebugAddress string

func (d RemoteDebugAddress) setForwarder(fwd *Forwarder) {
	fwd.remoteDebug = string(d)
}

// RemoteListenTimeout configures the amount of time the auto-started gopls
// daemon will wait with no client connections before shutting down.
type RemoteListenTimeout time.Duration

func (d RemoteListenTimeout) setForwarder(fwd *Forwarder) {
	fwd.remoteListenTimeout = time.Duration(d)
}

// RemoteLogfile configures the logfile location for the auto-started gopls
// daemon.
type RemoteLogfile string

func (l RemoteLogfile) setForwarder(fwd *Forwarder) {
	fwd.remoteLogfile = string(l)
}

// NewForwarder creates a new Forwarder, ready to forward connections to the
// remote server specified by network and addr.
func NewForwarder(network, addr string, opts ...ForwarderOption) *Forwarder {
	gp, err := os.Executable()
	if err != nil {
		log.Printf("error getting gopls path for forwarder: %v", err)
		gp = ""
	}

	fwd := &Forwarder{
		network:             network,
		addr:                addr,
		goplsPath:           gp,
		dialTimeout:         1 * time.Second,
		retries:             5,
		remoteLogfile:       "auto",
		remoteListenTimeout: 1 * time.Minute,
	}
	for _, opt := range opts {
		opt.setForwarder(fwd)
	}
	return fwd
}

// QueryServerState queries the server state of the current server.
func QueryServerState(ctx context.Context, network, address string) (*ServerState, error) {
	if network == AutoNetwork {
		gp, err := os.Executable()
		if err != nil {
			return nil, fmt.Errorf("getting gopls path: %w", err)
		}
		network, address = autoNetworkAddress(gp, address)
	}
	netConn, err := net.DialTimeout(network, address, 5*time.Second)
	if err != nil {
		return nil, fmt.Errorf("dialing remote: %w", err)
	}
	serverConn := jsonrpc2.NewConn(jsonrpc2.NewHeaderStream(netConn, netConn))
	go serverConn.Run(ctx, jsonrpc2.MethodNotFound)
	var state ServerState
	if err := protocol.Call(ctx, serverConn, sessionsMethod, nil, &state); err != nil {
		return nil, fmt.Errorf("querying server state: %w", err)
	}
	return &state, nil
}

// ServeStream dials the forwarder remote and binds the remote to serve the LSP
// on the incoming stream.
func (f *Forwarder) ServeStream(ctx context.Context, stream jsonrpc2.Stream) error {
	clientConn := jsonrpc2.NewConn(stream)
	client := protocol.ClientDispatcher(clientConn)

	netConn, err := f.connectToRemote(ctx)
	if err != nil {
		return fmt.Errorf("forwarder: connecting to remote: %w", err)
	}
	serverConn := jsonrpc2.NewConn(jsonrpc2.NewHeaderStream(netConn, netConn))
	server := protocol.ServerDispatcher(serverConn)

	// Forward between connections.
	g, ctx := errgroup.WithContext(ctx)
	g.Go(func() error {
		return serverConn.Run(ctx,
			protocol.Handlers(
				protocol.ClientHandler(client,
					jsonrpc2.MethodNotFound)))
	})
	// Don't run the clientConn yet, so that we can complete the handshake before
	// processing any client messages.

	// Do a handshake with the server instance to exchange debug information.
	index := atomic.AddInt64(&serverIndex, 1)
	serverID := strconv.FormatInt(index, 10)
	di := debug.GetInstance(ctx)
	var (
		hreq = handshakeRequest{
			ServerID:  serverID,
			GoplsPath: f.goplsPath,
		}
		hresp handshakeResponse
	)
	if di != nil {
		hreq.Logfile = di.Logfile
		hreq.DebugAddr = di.ListenedDebugAddress
	}
	if err := protocol.Call(ctx, serverConn, handshakeMethod, hreq, &hresp); err != nil {
		event.Error(ctx, "forwarder: gopls handshake failed", err)
	}
	if hresp.GoplsPath != f.goplsPath {
		event.Error(ctx, "", fmt.Errorf("forwarder: gopls path mismatch: forwarder is %q, remote is %q", f.goplsPath, hresp.GoplsPath))
	}
	if di != nil {
		di.State.AddServer(debugServer{
			debugInstance: debugInstance{
				id:           serverID,
				logfile:      hresp.Logfile,
				debugAddress: hresp.DebugAddr,
				goplsPath:    hresp.GoplsPath,
			},
			clientID: hresp.ClientID,
		})
	}
	g.Go(func() error {
		return clientConn.Run(ctx,
			protocol.Handlers(
				forwarderHandler(
					protocol.ServerHandler(server,
						jsonrpc2.MethodNotFound))))
	})

	return g.Wait()
}

func (f *Forwarder) connectToRemote(ctx context.Context) (net.Conn, error) {
	var (
		netConn          net.Conn
		err              error
		network, address = f.network, f.addr
	)
	if f.network == AutoNetwork {
		// f.network is overloaded to support a concept of 'automatic' addresses,
		// which signals that the gopls remote address should be automatically
		// derived.
		// So we need to resolve a real network and address here.
		network, address = autoNetworkAddress(f.goplsPath, f.addr)
	}
	// Attempt to verify that we own the remote. This is imperfect, but if we can
	// determine that the remote is owned by a different user, we should fail.
	ok, err := verifyRemoteOwnership(network, address)
	if err != nil {
		// If the ownership check itself failed, we fail open but log an error to
		// the user.
		event.Error(ctx, "unable to check daemon socket owner, failing open", err)
	} else if !ok {
		// We succesfully checked that the socket is not owned by us, we fail
		// closed.
		return nil, fmt.Errorf("socket %q is owned by a different user", address)
	}
	// Try dialing our remote once, in case it is already running.
	netConn, err = net.DialTimeout(network, address, f.dialTimeout)
	if err == nil {
		return netConn, nil
	}
	// If our remote is on the 'auto' network, start it if it doesn't exist.
	if f.network == AutoNetwork {
		if f.goplsPath == "" {
			return nil, fmt.Errorf("cannot auto-start remote: gopls path is unknown")
		}
		if network == "unix" {
			// Sometimes the socketfile isn't properly cleaned up when gopls shuts
			// down. Since we have already tried and failed to dial this address, it
			// should *usually* be safe to remove the socket before binding to the
			// address.
			// TODO(rfindley): there is probably a race here if multiple gopls
			// instances are simultaneously starting up.
			if _, err := os.Stat(address); err == nil {
				if err := os.Remove(address); err != nil {
					return nil, fmt.Errorf("removing remote socket file: %w", err)
				}
			}
		}
		args := []string{"serve",
			"-listen", fmt.Sprintf(`%s;%s`, network, address),
			"-listen.timeout", f.remoteListenTimeout.String(),
			"-logfile", f.remoteLogfile,
		}
		if f.remoteDebug != "" {
			args = append(args, "-debug", f.remoteDebug)
		}
		if err := startRemote(f.goplsPath, args...); err != nil {
			return nil, fmt.Errorf("startRemote(%q, %v): %w", f.goplsPath, args, err)
		}
	}

	// It can take some time for the newly started server to bind to our address,
	// so we retry for a bit.
	for retry := 0; retry < f.retries; retry++ {
		startDial := time.Now()
		netConn, err = net.DialTimeout(network, address, f.dialTimeout)
		if err == nil {
			return netConn, nil
		}
		event.Log(ctx, fmt.Sprintf("failed attempt #%d to connect to remote: %v\n", retry+2, err))
		// In case our failure was a fast-failure, ensure we wait at least
		// f.dialTimeout before trying again.
		if retry != f.retries-1 {
			time.Sleep(f.dialTimeout - time.Since(startDial))
		}
	}
	return nil, fmt.Errorf("dialing remote: %w", err)
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
func forwarderHandler(handler jsonrpc2.Handler) jsonrpc2.Handler {
	return func(ctx context.Context, reply jsonrpc2.Replier, r jsonrpc2.Request) error {
		// TODO(golang.org/issues/34111): we should more gracefully disconnect here,
		// once that process exists.
		if r.Method() == "exit" {
			ForwarderExitFunc(0)
			// reply nil here to consume the message: in
			// tests, ForwarderExitFunc may be overridden to something that doesn't
			// exit the process.
			return reply(ctx, nil, nil)
		}
		return handler(ctx, reply, r)
	}
}

// A handshakeRequest identifies a client to the LSP server.
type handshakeRequest struct {
	// ServerID is the ID of the server on the client. This should usually be 0.
	ServerID string `json:"serverID"`
	// Logfile is the location of the clients log file.
	Logfile string `json:"logfile"`
	// DebugAddr is the client debug address.
	DebugAddr string `json:"debugAddr"`
	// GoplsPath is the path to the Gopls binary running the current client
	// process.
	GoplsPath string `json:"goplsPath"`
}

// A handshakeResponse is returned by the LSP server to tell the LSP client
// information about its session.
type handshakeResponse struct {
	// ClientID is the ID of the client as seen on the server.
	ClientID string `json:"clientID"`
	// SessionID is the server session associated with the client.
	SessionID string `json:"sessionID"`
	// Logfile is the location of the server logs.
	Logfile string `json:"logfile"`
	// DebugAddr is the server debug address.
	DebugAddr string `json:"debugAddr"`
	// GoplsPath is the path to the Gopls binary running the current server
	// process.
	GoplsPath string `json:"goplsPath"`
}

// ClientSession identifies a current client LSP session on the server. Note
// that it looks similar to handshakeResposne, but in fact 'Logfile' and
// 'DebugAddr' now refer to the client.
type ClientSession struct {
	ClientID  string `json:"clientID"`
	SessionID string `json:"sessionID"`
	Logfile   string `json:"logfile"`
	DebugAddr string `json:"debugAddr"`
}

// ServerState holds information about the gopls daemon process, including its
// debug information and debug information of all of its current connected
// clients.
type ServerState struct {
	Logfile         string          `json:"logfile"`
	DebugAddr       string          `json:"debugAddr"`
	GoplsPath       string          `json:"goplsPath"`
	CurrentClientID string          `json:"currentClientID"`
	Clients         []ClientSession `json:"clients"`
}

const (
	handshakeMethod = "gopls/handshake"
	sessionsMethod  = "gopls/sessions"
)

func handshaker(client *debugClient, goplsPath string, handler jsonrpc2.Handler) jsonrpc2.Handler {
	return func(ctx context.Context, reply jsonrpc2.Replier, r jsonrpc2.Request) error {
		switch r.Method() {
		case handshakeMethod:
			var req handshakeRequest
			if err := json.Unmarshal(r.Params(), &req); err != nil {
				sendError(ctx, reply, err)
				return nil
			}
			client.debugAddress = req.DebugAddr
			client.logfile = req.Logfile
			client.serverID = req.ServerID
			client.goplsPath = req.GoplsPath
			resp := handshakeResponse{
				ClientID:  client.id,
				SessionID: cache.DebugSession{Session: client.session}.ID(),
				GoplsPath: goplsPath,
			}
			if di := debug.GetInstance(ctx); di != nil {
				resp.Logfile = di.Logfile
				resp.DebugAddr = di.ListenedDebugAddress
			}

			return reply(ctx, resp, nil)
		case sessionsMethod:
			resp := ServerState{
				GoplsPath:       goplsPath,
				CurrentClientID: client.ID(),
			}
			if di := debug.GetInstance(ctx); di != nil {
				resp.Logfile = di.Logfile
				resp.DebugAddr = di.ListenedDebugAddress
				for _, c := range di.State.Clients() {
					resp.Clients = append(resp.Clients, ClientSession{
						ClientID:  c.ID(),
						SessionID: c.Session().ID(),
						Logfile:   c.Logfile(),
						DebugAddr: c.DebugAddress(),
					})
				}
			}
			return reply(ctx, resp, nil)
		}
		return handler(ctx, reply, r)
	}
}

func sendError(ctx context.Context, reply jsonrpc2.Replier, err error) {
	err = fmt.Errorf("%w: %v", jsonrpc2.ErrParse, err)
	if err := reply(ctx, nil, err); err != nil {
		event.Error(ctx, "", err)
	}
}
