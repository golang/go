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

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/debug"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
)

// AutoNetwork is the pseudo network type used to signal that gopls should use
// automatic discovery to resolve a remote address.
const AutoNetwork = "auto"

// Unique identifiers for client/server.
var serverIndex int64

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

// ServeStream implements the jsonrpc2.StreamServer interface, by handling
// incoming streams using a new lsp server.
func (s *StreamServer) ServeStream(ctx context.Context, conn jsonrpc2.Conn) error {
	client := protocol.ClientDispatcher(conn)
	session := s.cache.NewSession(ctx)
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
	conn.Go(ctx,
		protocol.Handlers(
			handshaker(session, executable,
				protocol.ServerHandler(server,
					jsonrpc2.MethodNotFound))))
	<-conn.Done()
	return conn.Err()
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
	serverConn := jsonrpc2.NewConn(jsonrpc2.NewHeaderStream(netConn))
	serverConn.Go(ctx, jsonrpc2.MethodNotFound)
	var state ServerState
	if err := protocol.Call(ctx, serverConn, sessionsMethod, nil, &state); err != nil {
		return nil, fmt.Errorf("querying server state: %w", err)
	}
	return &state, nil
}

// ServeStream dials the forwarder remote and binds the remote to serve the LSP
// on the incoming stream.
func (f *Forwarder) ServeStream(ctx context.Context, clientConn jsonrpc2.Conn) error {
	client := protocol.ClientDispatcher(clientConn)

	netConn, err := f.connectToRemote(ctx)
	if err != nil {
		return fmt.Errorf("forwarder: connecting to remote: %w", err)
	}
	serverConn := jsonrpc2.NewConn(jsonrpc2.NewHeaderStream(netConn))
	server := protocol.ServerDispatcher(serverConn)

	// Forward between connections.
	serverConn.Go(ctx,
		protocol.Handlers(
			protocol.ClientHandler(client,
				jsonrpc2.MethodNotFound)))
	// Don't run the clientConn yet, so that we can complete the handshake before
	// processing any client messages.

	// Do a handshake with the server instance to exchange debug information.
	index := atomic.AddInt64(&serverIndex, 1)
	serverID := strconv.FormatInt(index, 10)
	var (
		hreq = handshakeRequest{
			ServerID:  serverID,
			GoplsPath: f.goplsPath,
		}
		hresp handshakeResponse
	)
	if di := debug.GetInstance(ctx); di != nil {
		hreq.Logfile = di.Logfile
		hreq.DebugAddr = di.ListenedDebugAddress
	}
	if err := protocol.Call(ctx, serverConn, handshakeMethod, hreq, &hresp); err != nil {
		event.Error(ctx, "forwarder: gopls handshake failed", err)
	}
	if hresp.GoplsPath != f.goplsPath {
		event.Error(ctx, "", fmt.Errorf("forwarder: gopls path mismatch: forwarder is %q, remote is %q", f.goplsPath, hresp.GoplsPath))
	}
	event.Log(ctx, "New server",
		tag.NewServer.Of(serverID),
		tag.Logfile.Of(hresp.Logfile),
		tag.DebugAddress.Of(hresp.DebugAddr),
		tag.GoplsPath.Of(hresp.GoplsPath),
		tag.ClientID.Of(hresp.SessionID),
	)
	clientConn.Go(ctx,
		protocol.Handlers(
			forwarderHandler(
				protocol.ServerHandler(server,
					jsonrpc2.MethodNotFound))))

	select {
	case <-serverConn.Done():
		clientConn.Close()
	case <-clientConn.Done():
		serverConn.Close()
	}

	err = serverConn.Err()
	if err == nil {
		err = clientConn.Err()
	}
	return err
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

// forwarderHandler intercepts 'exit' messages to prevent the shared gopls
// instance from exiting. In the future it may also intercept 'shutdown' to
// provide more graceful shutdown of the client connection.
func forwarderHandler(handler jsonrpc2.Handler) jsonrpc2.Handler {
	return func(ctx context.Context, reply jsonrpc2.Replier, r jsonrpc2.Request) error {
		// The gopls workspace environment defaults to the process environment in
		// which gopls daemon was started. To avoid discrepancies in Go environment
		// between the editor and daemon, inject any unset variables in `go env`
		// into the options sent by initialize.
		//
		// See also golang.org/issue/37830.
		if r.Method() == "initialize" {
			if newr, err := addGoEnvToInitializeRequest(ctx, r); err == nil {
				r = newr
			} else {
				log.Printf("unable to add local env to initialize request: %v", err)
			}
		}
		return handler(ctx, reply, r)
	}
}

// addGoEnvToInitializeRequest builds a new initialize request in which we set
// any environment variables output by `go env` and not already present in the
// request.
//
// It returns an error if r is not an initialize requst, or is otherwise
// malformed.
func addGoEnvToInitializeRequest(ctx context.Context, r jsonrpc2.Request) (jsonrpc2.Request, error) {
	var params protocol.ParamInitialize
	if err := json.Unmarshal(r.Params(), &params); err != nil {
		return nil, err
	}
	var opts map[string]interface{}
	switch v := params.InitializationOptions.(type) {
	case nil:
		opts = make(map[string]interface{})
	case map[string]interface{}:
		opts = v
	default:
		return nil, fmt.Errorf("unexpected type for InitializationOptions: %T", v)
	}
	envOpt, ok := opts["env"]
	if !ok {
		envOpt = make(map[string]interface{})
	}
	env, ok := envOpt.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf(`env option is %T, expected a map`, envOpt)
	}
	goenv, err := getGoEnv(ctx, env)
	if err != nil {
		return nil, err
	}
	for govar, value := range goenv {
		env[govar] = value
	}
	opts["env"] = env
	params.InitializationOptions = opts
	call, ok := r.(*jsonrpc2.Call)
	if !ok {
		return nil, fmt.Errorf("%T is not a *jsonrpc2.Call", r)
	}
	return jsonrpc2.NewCall(call.ID(), "initialize", params)
}

func getGoEnv(ctx context.Context, env map[string]interface{}) (map[string]string, error) {
	var runEnv []string
	for k, v := range env {
		runEnv = append(runEnv, fmt.Sprintf("%s=%s", k, v))
	}
	runner := gocommand.Runner{}
	output, err := runner.Run(ctx, gocommand.Invocation{
		Verb: "env",
		Args: []string{"-json"},
		Env:  runEnv,
	})
	if err != nil {
		return nil, err
	}
	envmap := make(map[string]string)
	if err := json.Unmarshal(output.Bytes(), &envmap); err != nil {
		return nil, err
	}
	return envmap, nil
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

func handshaker(session *cache.Session, goplsPath string, handler jsonrpc2.Handler) jsonrpc2.Handler {
	return func(ctx context.Context, reply jsonrpc2.Replier, r jsonrpc2.Request) error {
		switch r.Method() {
		case handshakeMethod:
			var req handshakeRequest
			if err := json.Unmarshal(r.Params(), &req); err != nil {
				sendError(ctx, reply, err)
				return nil
			}
			event.Log(ctx, "Handshake session update",
				cache.KeyUpdateSession.Of(session),
				tag.DebugAddress.Of(req.DebugAddr),
				tag.Logfile.Of(req.Logfile),
				tag.ServerID.Of(req.ServerID),
				tag.GoplsPath.Of(req.GoplsPath),
			)
			resp := handshakeResponse{
				SessionID: session.ID(),
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
				CurrentClientID: session.ID(),
			}
			if di := debug.GetInstance(ctx); di != nil {
				resp.Logfile = di.Logfile
				resp.DebugAddr = di.ListenedDebugAddress
				for _, c := range di.State.Clients() {
					resp.Clients = append(resp.Clients, ClientSession{
						SessionID: c.Session.ID(),
						Logfile:   c.Logfile,
						DebugAddr: c.DebugAddress,
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
