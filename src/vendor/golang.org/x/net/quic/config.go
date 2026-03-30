// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"crypto/tls"
	"log/slog"
	"math"
	"time"

	"golang.org/x/net/internal/quic/quicwire"
)

// A Config structure configures a QUIC endpoint.
// A Config must not be modified after it has been passed to a QUIC function.
// A Config may be reused; the quic package will also not modify it.
type Config struct {
	// TLSConfig is the endpoint's TLS configuration.
	// It must be non-nil and include at least one certificate or else set GetCertificate.
	TLSConfig *tls.Config

	// MaxBidiRemoteStreams limits the number of simultaneous bidirectional streams
	// a peer may open.
	// If zero, the default value of 100 is used.
	// If negative, the limit is zero.
	MaxBidiRemoteStreams int64

	// MaxUniRemoteStreams limits the number of simultaneous unidirectional streams
	// a peer may open.
	// If zero, the default value of 100 is used.
	// If negative, the limit is zero.
	MaxUniRemoteStreams int64

	// MaxStreamReadBufferSize is the maximum amount of data sent by the peer that a
	// stream will buffer for reading.
	// If zero, the default value of 1MiB is used.
	// If negative, the limit is zero.
	MaxStreamReadBufferSize int64

	// MaxStreamWriteBufferSize is the maximum amount of data a stream will buffer for
	// sending to the peer.
	// If zero, the default value of 1MiB is used.
	// If negative, the limit is zero.
	MaxStreamWriteBufferSize int64

	// MaxConnReadBufferSize is the maximum amount of data sent by the peer that a
	// connection will buffer for reading, across all streams.
	// If zero, the default value of 1MiB is used.
	// If negative, the limit is zero.
	MaxConnReadBufferSize int64

	// RequireAddressValidation may be set to true to enable address validation
	// of client connections prior to starting the handshake.
	//
	// Enabling this setting reduces the amount of work packets with spoofed
	// source address information can cause a server to perform,
	// at the cost of increased handshake latency.
	RequireAddressValidation bool

	// StatelessResetKey is used to provide stateless reset of connections.
	// A restart may leave an endpoint without access to the state of
	// existing connections. Stateless reset permits an endpoint to respond
	// to a packet for a connection it does not recognize.
	//
	// This field should be filled with random bytes.
	// The contents should remain stable across restarts,
	// to permit an endpoint to send a reset for
	// connections created before a restart.
	//
	// The contents of the StatelessResetKey should not be exposed.
	// An attacker can use knowledge of this field's value to
	// reset existing connections.
	//
	// If this field is left as zero, stateless reset is disabled.
	StatelessResetKey [32]byte

	// HandshakeTimeout is the maximum time in which a connection handshake must complete.
	// If zero, the default of 10 seconds is used.
	// If negative, there is no handshake timeout.
	HandshakeTimeout time.Duration

	// MaxIdleTimeout is the maximum time after which an idle connection will be closed.
	// If zero, the default of 30 seconds is used.
	// If negative, idle connections are never closed.
	//
	// The idle timeout for a connection is the minimum of the maximum idle timeouts
	// of the endpoints.
	MaxIdleTimeout time.Duration

	// KeepAlivePeriod is the time after which a packet will be sent to keep
	// an idle connection alive.
	// If zero, keep alive packets are not sent.
	// If greater than zero, the keep alive period is the smaller of KeepAlivePeriod and
	// half the connection idle timeout.
	KeepAlivePeriod time.Duration

	// QLogLogger receives qlog events.
	//
	// Events currently correspond to the definitions in draft-ietf-qlog-quic-events-03.
	// This is not the latest version of the draft, but is the latest version supported
	// by common event log viewers as of the time this paragraph was written.
	//
	// The qlog package contains a slog.Handler which serializes qlog events
	// to a standard JSON representation.
	QLogLogger *slog.Logger
}

// Clone returns a shallow clone of c, or nil if c is nil.
// It is safe to clone a [Config] that is being used concurrently by a QUIC endpoint.
func (c *Config) Clone() *Config {
	n := *c
	return &n
}

func configDefault[T ~int64](v, def, limit T) T {
	switch {
	case v == 0:
		return def
	case v < 0:
		return 0
	default:
		return min(v, limit)
	}
}

func (c *Config) maxBidiRemoteStreams() int64 {
	return configDefault(c.MaxBidiRemoteStreams, 100, maxStreamsLimit)
}

func (c *Config) maxUniRemoteStreams() int64 {
	return configDefault(c.MaxUniRemoteStreams, 100, maxStreamsLimit)
}

func (c *Config) maxStreamReadBufferSize() int64 {
	return configDefault(c.MaxStreamReadBufferSize, 1<<20, quicwire.MaxVarint)
}

func (c *Config) maxStreamWriteBufferSize() int64 {
	return configDefault(c.MaxStreamWriteBufferSize, 1<<20, quicwire.MaxVarint)
}

func (c *Config) maxConnReadBufferSize() int64 {
	return configDefault(c.MaxConnReadBufferSize, 1<<20, quicwire.MaxVarint)
}

func (c *Config) handshakeTimeout() time.Duration {
	return configDefault(c.HandshakeTimeout, defaultHandshakeTimeout, math.MaxInt64)
}

func (c *Config) maxIdleTimeout() time.Duration {
	return configDefault(c.MaxIdleTimeout, defaultMaxIdleTimeout, math.MaxInt64)
}

func (c *Config) keepAlivePeriod() time.Duration {
	return configDefault(c.KeepAlivePeriod, defaultKeepAlivePeriod, math.MaxInt64)
}
