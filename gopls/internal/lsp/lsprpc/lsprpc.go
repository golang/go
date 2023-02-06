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
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/tools/gopls/internal/lsp"
	"golang.org/x/tools/gopls/internal/lsp/cache"
	"golang.org/x/tools/gopls/internal/lsp/command"
	"golang.org/x/tools/gopls/internal/lsp/debug"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/tag"
	"golang.org/x/tools/internal/jsonrpc2"
)

// Unique identifiers for client/server.
var serverIndex int64

// The StreamServer type is a jsonrpc2.StreamServer that handles incoming
// streams as a new LSP session, using a shared cache.
type StreamServer struct {
	cache *cache.Cache
	// daemon controls whether or not to log new connections.
	daemon bool

	// optionsOverrides is passed to newly created sessions.
	optionsOverrides func(*source.Options)

	// serverForTest may be set to a test fake for testing.
	serverForTest protocol.Server
}

// NewStreamServer creates a StreamServer using the shared cache. If
// withTelemetry is true, each session is instrumented with telemetry that
// records RPC statistics.
func NewStreamServer(cache *cache.Cache, daemon bool, optionsFunc func(*source.Options)) *StreamServer {
	return &StreamServer{cache: cache, daemon: daemon, optionsOverrides: optionsFunc}
}

func (s *StreamServer) Binder() *ServerBinder {
	newServer := func(ctx context.Context, client protocol.ClientCloser) protocol.Server {
		session := cache.NewSession(ctx, s.cache, s.optionsOverrides)
		server := s.serverForTest
		if server == nil {
			server = lsp.NewServer(session, client)
			if instance := debug.GetInstance(ctx); instance != nil {
				instance.AddService(server, session)
			}
		}
		return server
	}
	return NewServerBinder(newServer)
}

// ServeStream implements the jsonrpc2.StreamServer interface, by handling
// incoming streams using a new lsp server.
func (s *StreamServer) ServeStream(ctx context.Context, conn jsonrpc2.Conn) error {
	client := protocol.ClientDispatcher(conn)
	session := cache.NewSession(ctx, s.cache, s.optionsOverrides)
	server := s.serverForTest
	if server == nil {
		server = lsp.NewServer(session, client)
		if instance := debug.GetInstance(ctx); instance != nil {
			instance.AddService(server, session)
		}
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
			handshaker(session, executable, s.daemon,
				protocol.ServerHandler(server,
					jsonrpc2.MethodNotFound))))
	if s.daemon {
		log.Printf("Session %s: connected", session.ID())
		defer log.Printf("Session %s: exited", session.ID())
	}
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
	dialer *AutoDialer

	mu sync.Mutex
	// Hold on to the server connection so that we can redo the handshake if any
	// information changes.
	serverConn jsonrpc2.Conn
	serverID   string
}

// NewForwarder creates a new Forwarder, ready to forward connections to the
// remote server specified by rawAddr. If provided and rawAddr indicates an
// 'automatic' address (starting with 'auto;'), argFunc may be used to start a
// remote server for the auto-discovered address.
func NewForwarder(rawAddr string, argFunc func(network, address string) []string) (*Forwarder, error) {
	dialer, err := NewAutoDialer(rawAddr, argFunc)
	if err != nil {
		return nil, err
	}
	fwd := &Forwarder{
		dialer: dialer,
	}
	return fwd, nil
}

// QueryServerState queries the server state of the current server.
func QueryServerState(ctx context.Context, addr string) (*ServerState, error) {
	serverConn, err := dialRemote(ctx, addr)
	if err != nil {
		return nil, err
	}
	var state ServerState
	if err := protocol.Call(ctx, serverConn, sessionsMethod, nil, &state); err != nil {
		return nil, fmt.Errorf("querying server state: %w", err)
	}
	return &state, nil
}

// dialRemote is used for making calls into the gopls daemon. addr should be a
// URL, possibly on the synthetic 'auto' network (e.g. tcp://..., unix://...,
// or auto://...).
func dialRemote(ctx context.Context, addr string) (jsonrpc2.Conn, error) {
	network, address := ParseAddr(addr)
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
	return serverConn, nil
}

func ExecuteCommand(ctx context.Context, addr string, id string, request, result interface{}) error {
	serverConn, err := dialRemote(ctx, addr)
	if err != nil {
		return err
	}
	args, err := command.MarshalArgs(request)
	if err != nil {
		return err
	}
	params := protocol.ExecuteCommandParams{
		Command:   id,
		Arguments: args,
	}
	return protocol.Call(ctx, serverConn, "workspace/executeCommand", params, result)
}

// ServeStream dials the forwarder remote and binds the remote to serve the LSP
// on the incoming stream.
func (f *Forwarder) ServeStream(ctx context.Context, clientConn jsonrpc2.Conn) error {
	client := protocol.ClientDispatcher(clientConn)

	netConn, err := f.dialer.dialNet(ctx)
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
	f.mu.Lock()
	f.serverConn = serverConn
	f.serverID = strconv.FormatInt(index, 10)
	f.mu.Unlock()
	f.handshake(ctx)
	clientConn.Go(ctx,
		protocol.Handlers(
			f.handler(
				protocol.ServerHandler(server,
					jsonrpc2.MethodNotFound))))

	select {
	case <-serverConn.Done():
		clientConn.Close()
	case <-clientConn.Done():
		serverConn.Close()
	}

	err = nil
	if serverConn.Err() != nil {
		err = fmt.Errorf("remote disconnected: %v", serverConn.Err())
	} else if clientConn.Err() != nil {
		err = fmt.Errorf("client disconnected: %v", clientConn.Err())
	}
	event.Log(ctx, fmt.Sprintf("forwarder: exited with error: %v", err))
	return err
}

// TODO(rfindley): remove this handshaking in favor of middleware.
func (f *Forwarder) handshake(ctx context.Context) {
	// This call to os.Executable is redundant, and will be eliminated by the
	// transition to the V2 API.
	goplsPath, err := os.Executable()
	if err != nil {
		event.Error(ctx, "getting executable for handshake", err)
		goplsPath = ""
	}
	var (
		hreq = handshakeRequest{
			ServerID:  f.serverID,
			GoplsPath: goplsPath,
		}
		hresp handshakeResponse
	)
	if di := debug.GetInstance(ctx); di != nil {
		hreq.Logfile = di.Logfile
		hreq.DebugAddr = di.ListenedDebugAddress()
	}
	if err := protocol.Call(ctx, f.serverConn, handshakeMethod, hreq, &hresp); err != nil {
		// TODO(rfindley): at some point in the future we should return an error
		// here.  Handshakes have become functional in nature.
		event.Error(ctx, "forwarder: gopls handshake failed", err)
	}
	if hresp.GoplsPath != goplsPath {
		event.Error(ctx, "", fmt.Errorf("forwarder: gopls path mismatch: forwarder is %q, remote is %q", goplsPath, hresp.GoplsPath))
	}
	event.Log(ctx, "New server",
		tag.NewServer.Of(f.serverID),
		tag.Logfile.Of(hresp.Logfile),
		tag.DebugAddress.Of(hresp.DebugAddr),
		tag.GoplsPath.Of(hresp.GoplsPath),
		tag.ClientID.Of(hresp.SessionID),
	)
}

func ConnectToRemote(ctx context.Context, addr string) (net.Conn, error) {
	dialer, err := NewAutoDialer(addr, nil)
	if err != nil {
		return nil, err
	}
	return dialer.dialNet(ctx)
}

// handler intercepts messages to the daemon to enrich them with local
// information.
func (f *Forwarder) handler(handler jsonrpc2.Handler) jsonrpc2.Handler {
	return func(ctx context.Context, reply jsonrpc2.Replier, r jsonrpc2.Request) error {
		// Intercept certain messages to add special handling.
		switch r.Method() {
		case "initialize":
			if newr, err := addGoEnvToInitializeRequest(ctx, r); err == nil {
				r = newr
			} else {
				log.Printf("unable to add local env to initialize request: %v", err)
			}
		case "workspace/executeCommand":
			var params protocol.ExecuteCommandParams
			if err := json.Unmarshal(r.Params(), &params); err == nil {
				if params.Command == command.StartDebugging.ID() {
					var args command.DebuggingArgs
					if err := command.UnmarshalArgs(params.Arguments, &args); err == nil {
						reply = f.replyWithDebugAddress(ctx, reply, args)
					} else {
						event.Error(ctx, "unmarshaling debugging args", err)
					}
				}
			} else {
				event.Error(ctx, "intercepting executeCommand request", err)
			}
		}
		// The gopls workspace environment defaults to the process environment in
		// which gopls daemon was started. To avoid discrepancies in Go environment
		// between the editor and daemon, inject any unset variables in `go env`
		// into the options sent by initialize.
		//
		// See also golang.org/issue/37830.
		return handler(ctx, reply, r)
	}
}

// addGoEnvToInitializeRequest builds a new initialize request in which we set
// any environment variables output by `go env` and not already present in the
// request.
//
// It returns an error if r is not an initialize request, or is otherwise
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
	// We don't want to propagate GOWORK unless explicitly set since that could mess with
	// path inference during cmd/go invocations, see golang/go#51825.
	_, goworkSet := os.LookupEnv("GOWORK")
	for govar, value := range goenv {
		if govar == "GOWORK" && !goworkSet {
			continue
		}
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

func (f *Forwarder) replyWithDebugAddress(outerCtx context.Context, r jsonrpc2.Replier, args command.DebuggingArgs) jsonrpc2.Replier {
	di := debug.GetInstance(outerCtx)
	if di == nil {
		event.Log(outerCtx, "no debug instance to start")
		return r
	}
	return func(ctx context.Context, result interface{}, outerErr error) error {
		if outerErr != nil {
			return r(ctx, result, outerErr)
		}
		// Enrich the result with our own debugging information. Since we're an
		// intermediary, the jsonrpc2 package has deserialized the result into
		// maps, by default. Re-do the unmarshalling.
		raw, err := json.Marshal(result)
		if err != nil {
			event.Error(outerCtx, "marshaling intermediate command result", err)
			return r(ctx, result, err)
		}
		var modified command.DebuggingResult
		if err := json.Unmarshal(raw, &modified); err != nil {
			event.Error(outerCtx, "unmarshaling intermediate command result", err)
			return r(ctx, result, err)
		}
		addr := args.Addr
		if addr == "" {
			addr = "localhost:0"
		}
		addr, err = di.Serve(outerCtx, addr)
		if err != nil {
			event.Error(outerCtx, "starting debug server", err)
			return r(ctx, result, outerErr)
		}
		urls := []string{"http://" + addr}
		modified.URLs = append(urls, modified.URLs...)
		go f.handshake(ctx)
		return r(ctx, modified, nil)
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

func handshaker(session *cache.Session, goplsPath string, logHandshakes bool, handler jsonrpc2.Handler) jsonrpc2.Handler {
	return func(ctx context.Context, reply jsonrpc2.Replier, r jsonrpc2.Request) error {
		switch r.Method() {
		case handshakeMethod:
			// We log.Printf in this handler, rather than event.Log when we want logs
			// to go to the daemon log rather than being reflected back to the
			// client.
			var req handshakeRequest
			if err := json.Unmarshal(r.Params(), &req); err != nil {
				if logHandshakes {
					log.Printf("Error processing handshake for session %s: %v", session.ID(), err)
				}
				sendError(ctx, reply, err)
				return nil
			}
			if logHandshakes {
				log.Printf("Session %s: got handshake. Logfile: %q, Debug addr: %q", session.ID(), req.Logfile, req.DebugAddr)
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
				resp.DebugAddr = di.ListenedDebugAddress()
			}
			return reply(ctx, resp, nil)

		case sessionsMethod:
			resp := ServerState{
				GoplsPath:       goplsPath,
				CurrentClientID: session.ID(),
			}
			if di := debug.GetInstance(ctx); di != nil {
				resp.Logfile = di.Logfile
				resp.DebugAddr = di.ListenedDebugAddress()
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
	err = fmt.Errorf("%v: %w", err, jsonrpc2.ErrParse)
	if err := reply(ctx, nil, err); err != nil {
		event.Error(ctx, "", err)
	}
}

// ParseAddr parses the address of a gopls remote.
// TODO(rFindley): further document this syntax, and allow URI-style remote
// addresses such as "auto://...".
func ParseAddr(listen string) (network string, address string) {
	// Allow passing just -remote=auto, as a shorthand for using automatic remote
	// resolution.
	if listen == AutoNetwork {
		return AutoNetwork, ""
	}
	if parts := strings.SplitN(listen, ";", 2); len(parts) == 2 {
		return parts[0], parts[1]
	}
	return "tcp", listen
}
