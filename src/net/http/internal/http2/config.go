// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"math"
	"time"
)

// Config must be kept in sync with net/http.HTTP2Config.
type Config struct {
	MaxConcurrentStreams          int
	StrictMaxConcurrentRequests   bool
	MaxDecoderHeaderTableSize     int
	MaxEncoderHeaderTableSize     int
	MaxReadFrameSize              int
	MaxReceiveBufferPerConnection int
	MaxReceiveBufferPerStream     int
	SendPingTimeout               time.Duration
	PingTimeout                   time.Duration
	WriteByteTimeout              time.Duration
	PermitProhibitedCipherSuites  bool
	CountError                    func(errType string)
}

func configFromServer(h1 ServerConfig, h2 *Server) Config {
	conf := Config{
		MaxConcurrentStreams:          int(h2.MaxConcurrentStreams),
		MaxEncoderHeaderTableSize:     int(h2.MaxEncoderHeaderTableSize),
		MaxDecoderHeaderTableSize:     int(h2.MaxDecoderHeaderTableSize),
		MaxReadFrameSize:              int(h2.MaxReadFrameSize),
		MaxReceiveBufferPerConnection: int(h2.MaxUploadBufferPerConnection),
		MaxReceiveBufferPerStream:     int(h2.MaxUploadBufferPerStream),
		SendPingTimeout:               h2.ReadIdleTimeout,
		PingTimeout:                   h2.PingTimeout,
		WriteByteTimeout:              h2.WriteByteTimeout,
		PermitProhibitedCipherSuites:  h2.PermitProhibitedCipherSuites,
		CountError:                    h2.CountError,
	}
	fillNetHTTPConfig(&conf, h1.HTTP2Config())
	setConfigDefaults(&conf, true)
	return conf
}

func configFromTransport(h2 *Transport) Config {
	conf := Config{
		MaxEncoderHeaderTableSize: int(h2.MaxEncoderHeaderTableSize),
		MaxDecoderHeaderTableSize: int(h2.MaxDecoderHeaderTableSize),
		MaxReadFrameSize:          int(h2.MaxReadFrameSize),
		SendPingTimeout:           h2.ReadIdleTimeout,
		PingTimeout:               h2.PingTimeout,
		WriteByteTimeout:          h2.WriteByteTimeout,
	}

	// Unlike most config fields, where out-of-range values revert to the default,
	// Transport.MaxReadFrameSize clips.
	if conf.MaxReadFrameSize < minMaxFrameSize {
		conf.MaxReadFrameSize = minMaxFrameSize
	} else if conf.MaxReadFrameSize > maxFrameSize {
		conf.MaxReadFrameSize = maxFrameSize
	}

	if h2.t1 != nil {
		fillNetHTTPConfig(&conf, h2.t1.HTTP2Config())
	}

	setConfigDefaults(&conf, false)
	return conf
}

func setDefault[T ~int | ~int32 | ~uint32 | ~int64](v *T, minval, maxval, defval T) {
	if *v < minval || *v > maxval {
		*v = defval
	}
}

func setConfigDefaults(conf *Config, server bool) {
	setDefault(&conf.MaxConcurrentStreams, 1, math.MaxInt32, defaultMaxStreams)
	setDefault(&conf.MaxEncoderHeaderTableSize, 1, math.MaxInt32, initialHeaderTableSize)
	setDefault(&conf.MaxDecoderHeaderTableSize, 1, math.MaxInt32, initialHeaderTableSize)
	if server {
		setDefault(&conf.MaxReceiveBufferPerConnection, initialWindowSize, math.MaxInt32, 1<<20)
	} else {
		setDefault(&conf.MaxReceiveBufferPerConnection, initialWindowSize, math.MaxInt32, transportDefaultConnFlow)
	}
	if server {
		setDefault(&conf.MaxReceiveBufferPerStream, 1, math.MaxInt32, 1<<20)
	} else {
		setDefault(&conf.MaxReceiveBufferPerStream, 1, math.MaxInt32, transportDefaultStreamFlow)
	}
	setDefault(&conf.MaxReadFrameSize, minMaxFrameSize, maxFrameSize, defaultMaxReadFrameSize)
	setDefault(&conf.PingTimeout, 1, math.MaxInt64, 15*time.Second)
}

// adjustHTTP1MaxHeaderSize converts a limit in bytes on the size of an HTTP/1 header
// to an HTTP/2 MAX_HEADER_LIST_SIZE value.
func adjustHTTP1MaxHeaderSize(n int64) int64 {
	// http2's count is in a slightly different unit and includes 32 bytes per pair.
	// So, take the net/http.Server value and pad it up a bit, assuming 10 headers.
	const perFieldOverhead = 32 // per http2 spec
	const typicalHeaders = 10   // conservative
	return n + typicalHeaders*perFieldOverhead
}

func fillNetHTTPConfig(conf *Config, h2 Config) {
	if h2.MaxConcurrentStreams != 0 {
		conf.MaxConcurrentStreams = h2.MaxConcurrentStreams
	}
	if h2.StrictMaxConcurrentRequests {
		conf.StrictMaxConcurrentRequests = true
	}
	if h2.MaxEncoderHeaderTableSize != 0 {
		conf.MaxEncoderHeaderTableSize = h2.MaxEncoderHeaderTableSize
	}
	if h2.MaxDecoderHeaderTableSize != 0 {
		conf.MaxDecoderHeaderTableSize = h2.MaxDecoderHeaderTableSize
	}
	if h2.MaxConcurrentStreams != 0 {
		conf.MaxConcurrentStreams = h2.MaxConcurrentStreams
	}
	if h2.MaxReadFrameSize != 0 {
		conf.MaxReadFrameSize = h2.MaxReadFrameSize
	}
	if h2.MaxReceiveBufferPerConnection != 0 {
		conf.MaxReceiveBufferPerConnection = h2.MaxReceiveBufferPerConnection
	}
	if h2.MaxReceiveBufferPerStream != 0 {
		conf.MaxReceiveBufferPerStream = h2.MaxReceiveBufferPerStream
	}
	if h2.SendPingTimeout != 0 {
		conf.SendPingTimeout = h2.SendPingTimeout
	}
	if h2.PingTimeout != 0 {
		conf.PingTimeout = h2.PingTimeout
	}
	if h2.WriteByteTimeout != 0 {
		conf.WriteByteTimeout = h2.WriteByteTimeout
	}
	if h2.PermitProhibitedCipherSuites {
		conf.PermitProhibitedCipherSuites = true
	}
	if h2.CountError != nil {
		conf.CountError = h2.CountError
	}
}
