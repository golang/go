// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonrpc2

import (
	"context"
	"io"
	"net"
	"os"
)

// This file contains implementations of the transport primitives that use the standard network
// package.

// NetListenOptions is the optional arguments to the NetListen function.
type NetListenOptions struct {
	NetListenConfig net.ListenConfig
	NetDialer       net.Dialer
}

// NetListener returns a new Listener that listens on a socket using the net package.
func NetListener(ctx context.Context, network, address string, options NetListenOptions) (Listener, error) {
	ln, err := options.NetListenConfig.Listen(ctx, network, address)
	if err != nil {
		return nil, err
	}
	return &netListener{net: ln}, nil
}

// netListener is the implementation of Listener for connections made using the net package.
type netListener struct {
	net net.Listener
}

// Accept blocks waiting for an incoming connection to the listener.
func (l *netListener) Accept(context.Context) (io.ReadWriteCloser, error) {
	return l.net.Accept()
}

// Close will cause the listener to stop listening. It will not close any connections that have
// already been accepted.
func (l *netListener) Close() error {
	addr := l.net.Addr()
	err := l.net.Close()
	if addr.Network() == "unix" {
		rerr := os.Remove(addr.String())
		if rerr != nil && err == nil {
			err = rerr
		}
	}
	return err
}

// Dialer returns a dialer that can be used to connect to the listener.
func (l *netListener) Dialer() Dialer {
	return NetDialer(l.net.Addr().Network(), l.net.Addr().String(), net.Dialer{})
}

// NetDialer returns a Dialer using the supplied standard network dialer.
func NetDialer(network, address string, nd net.Dialer) Dialer {
	return &netDialer{
		network: network,
		address: address,
		dialer:  nd,
	}
}

type netDialer struct {
	network string
	address string
	dialer  net.Dialer
}

func (n *netDialer) Dial(ctx context.Context) (io.ReadWriteCloser, error) {
	return n.dialer.DialContext(ctx, n.network, n.address)
}

// NetPipeListener returns a new Listener that listens using net.Pipe.
// It is only possibly to connect to it using the Dialer returned by the
// Dialer method, each call to that method will generate a new pipe the other
// side of which will be returned from the Accept call.
func NetPipeListener(ctx context.Context) (Listener, error) {
	return &netPiper{
		done:   make(chan struct{}),
		dialed: make(chan io.ReadWriteCloser),
	}, nil
}

// netPiper is the implementation of Listener build on top of net.Pipes.
type netPiper struct {
	done   chan struct{}
	dialed chan io.ReadWriteCloser
}

// Accept blocks waiting for an incoming connection to the listener.
func (l *netPiper) Accept(context.Context) (io.ReadWriteCloser, error) {
	// Block until the pipe is dialed or the listener is closed,
	// preferring the latter if already closed at the start of Accept.
	select {
	case <-l.done:
		return nil, errClosed
	default:
	}
	select {
	case rwc := <-l.dialed:
		return rwc, nil
	case <-l.done:
		return nil, errClosed
	}
}

// Close will cause the listener to stop listening. It will not close any connections that have
// already been accepted.
func (l *netPiper) Close() error {
	// unblock any accept calls that are pending
	close(l.done)
	return nil
}

func (l *netPiper) Dialer() Dialer {
	return l
}

func (l *netPiper) Dial(ctx context.Context) (io.ReadWriteCloser, error) {
	client, server := net.Pipe()

	select {
	case l.dialed <- server:
		return client, nil

	case <-l.done:
		client.Close()
		server.Close()
		return nil, errClosed
	}
}
