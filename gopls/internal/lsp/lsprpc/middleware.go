// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsprpc

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"golang.org/x/tools/internal/event"
	jsonrpc2_v2 "golang.org/x/tools/internal/jsonrpc2_v2"
)

// Metadata holds arbitrary data transferred between jsonrpc2 peers.
type Metadata map[string]interface{}

// PeerInfo holds information about a peering between jsonrpc2 servers.
type PeerInfo struct {
	// RemoteID is the identity of the current server on its peer.
	RemoteID int64

	// LocalID is the identity of the peer on the server.
	LocalID int64

	// IsClient reports whether the peer is a client. If false, the peer is a
	// server.
	IsClient bool

	// Metadata holds arbitrary information provided by the peer.
	Metadata Metadata
}

// Handshaker handles both server and client handshaking over jsonrpc2. To
// instrument server-side handshaking, use Handshaker.Middleware. To instrument
// client-side handshaking, call Handshaker.ClientHandshake for any new
// client-side connections.
type Handshaker struct {
	// Metadata will be shared with peers via handshaking.
	Metadata Metadata

	mu     sync.Mutex
	prevID int64
	peers  map[int64]PeerInfo
}

// Peers returns the peer info this handshaker knows about by way of either the
// server-side handshake middleware, or client-side handshakes.
func (h *Handshaker) Peers() []PeerInfo {
	h.mu.Lock()
	defer h.mu.Unlock()

	var c []PeerInfo
	for _, v := range h.peers {
		c = append(c, v)
	}
	return c
}

// Middleware is a jsonrpc2 middleware function to augment connection binding
// to handle the handshake method, and record disconnections.
func (h *Handshaker) Middleware(inner jsonrpc2_v2.Binder) jsonrpc2_v2.Binder {
	return BinderFunc(func(ctx context.Context, conn *jsonrpc2_v2.Connection) jsonrpc2_v2.ConnectionOptions {
		opts := inner.Bind(ctx, conn)

		localID := h.nextID()
		info := &PeerInfo{
			RemoteID: localID,
			Metadata: h.Metadata,
		}

		// Wrap the delegated handler to accept the handshake.
		delegate := opts.Handler
		opts.Handler = jsonrpc2_v2.HandlerFunc(func(ctx context.Context, req *jsonrpc2_v2.Request) (interface{}, error) {
			if req.Method == handshakeMethod {
				var peerInfo PeerInfo
				if err := json.Unmarshal(req.Params, &peerInfo); err != nil {
					return nil, fmt.Errorf("%w: unmarshaling client info: %v", jsonrpc2_v2.ErrInvalidParams, err)
				}
				peerInfo.LocalID = localID
				peerInfo.IsClient = true
				h.recordPeer(peerInfo)
				return info, nil
			}
			return delegate.Handle(ctx, req)
		})

		// Record the dropped client.
		go h.cleanupAtDisconnect(conn, localID)

		return opts
	})
}

// ClientHandshake performs a client-side handshake with the server at the
// other end of conn, recording the server's peer info and watching for conn's
// disconnection.
func (h *Handshaker) ClientHandshake(ctx context.Context, conn *jsonrpc2_v2.Connection) {
	localID := h.nextID()
	info := &PeerInfo{
		RemoteID: localID,
		Metadata: h.Metadata,
	}

	call := conn.Call(ctx, handshakeMethod, info)
	var serverInfo PeerInfo
	if err := call.Await(ctx, &serverInfo); err != nil {
		event.Error(ctx, "performing handshake", err)
		return
	}
	serverInfo.LocalID = localID
	h.recordPeer(serverInfo)

	go h.cleanupAtDisconnect(conn, localID)
}

func (h *Handshaker) nextID() int64 {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.prevID++
	return h.prevID
}

func (h *Handshaker) cleanupAtDisconnect(conn *jsonrpc2_v2.Connection, peerID int64) {
	conn.Wait()

	h.mu.Lock()
	defer h.mu.Unlock()
	delete(h.peers, peerID)
}

func (h *Handshaker) recordPeer(info PeerInfo) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.peers == nil {
		h.peers = make(map[int64]PeerInfo)
	}
	h.peers[info.LocalID] = info
}
