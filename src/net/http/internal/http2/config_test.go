// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

package http2_test

import (
	"net/http"
	"testing"
	"time"

	. "golang.org/x/net/http2"
)

func TestConfigServerSettings(t *testing.T) { synctestTest(t, testConfigServerSettings) }
func testConfigServerSettings(t testing.TB) {
	config := &http.HTTP2Config{
		MaxConcurrentStreams:          1,
		MaxDecoderHeaderTableSize:     1<<20 + 2,
		MaxEncoderHeaderTableSize:     1<<20 + 3,
		MaxReadFrameSize:              1<<20 + 4,
		MaxReceiveBufferPerConnection: 64<<10 + 5,
		MaxReceiveBufferPerStream:     64<<10 + 6,
	}
	const maxHeaderBytes = 4096 + 7
	st := newServerTester(t, nil, func(s *http.Server) {
		s.MaxHeaderBytes = maxHeaderBytes
		s.HTTP2 = config
	})
	st.writePreface()
	st.writeSettings()
	st.wantSettings(map[SettingID]uint32{
		SettingMaxConcurrentStreams: uint32(config.MaxConcurrentStreams),
		SettingHeaderTableSize:      uint32(config.MaxDecoderHeaderTableSize),
		SettingInitialWindowSize:    uint32(config.MaxReceiveBufferPerStream),
		SettingMaxFrameSize:         uint32(config.MaxReadFrameSize),
		SettingMaxHeaderListSize:    maxHeaderBytes + (32 * 10),
	})
}

func TestConfigTransportSettings(t *testing.T) { synctestTest(t, testConfigTransportSettings) }
func testConfigTransportSettings(t testing.TB) {
	config := &http.HTTP2Config{
		MaxConcurrentStreams:          1, // ignored by Transport
		MaxDecoderHeaderTableSize:     1<<20 + 2,
		MaxEncoderHeaderTableSize:     1<<20 + 3,
		MaxReadFrameSize:              1<<20 + 4,
		MaxReceiveBufferPerConnection: 64<<10 + 5,
		MaxReceiveBufferPerStream:     64<<10 + 6,
	}
	const maxHeaderBytes = 4096 + 7
	tc := newTestClientConn(t, func(tr *http.Transport) {
		tr.HTTP2 = config
		tr.MaxResponseHeaderBytes = maxHeaderBytes
	})
	tc.wantSettings(map[SettingID]uint32{
		SettingHeaderTableSize:   uint32(config.MaxDecoderHeaderTableSize),
		SettingInitialWindowSize: uint32(config.MaxReceiveBufferPerStream),
		SettingMaxFrameSize:      uint32(config.MaxReadFrameSize),
		SettingMaxHeaderListSize: maxHeaderBytes + (32 * 10),
	})
	tc.wantWindowUpdate(0, uint32(config.MaxReceiveBufferPerConnection))
}

func TestConfigPingTimeoutServer(t *testing.T) { synctestTest(t, testConfigPingTimeoutServer) }
func testConfigPingTimeoutServer(t testing.TB) {
	st := newServerTester(t, func(w http.ResponseWriter, r *http.Request) {
	}, func(s *Server) {
		s.ReadIdleTimeout = 2 * time.Second
		s.PingTimeout = 3 * time.Second
	})
	st.greet()

	time.Sleep(2 * time.Second)
	_ = readFrame[*PingFrame](t, st)
	time.Sleep(3 * time.Second)
	st.wantClosed()
}

func TestConfigPingTimeoutTransport(t *testing.T) { synctestTest(t, testConfigPingTimeoutTransport) }
func testConfigPingTimeoutTransport(t testing.TB) {
	tc := newTestClientConn(t, func(tr *Transport) {
		tr.ReadIdleTimeout = 2 * time.Second
		tr.PingTimeout = 3 * time.Second
	})
	tc.greet()

	req, _ := http.NewRequest("GET", "https://dummy.tld/", nil)
	rt := tc.roundTrip(req)
	tc.wantFrameType(FrameHeaders)

	time.Sleep(2 * time.Second)
	tc.wantFrameType(FramePing)
	time.Sleep(3 * time.Second)
	err := rt.err()
	if err == nil {
		t.Fatalf("expected connection to close")
	}
}
