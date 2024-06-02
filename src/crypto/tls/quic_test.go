// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bytes"
	"context"
	"errors"
	"reflect"
	"testing"
)

type testQUICConn struct {
	t                 *testing.T
	conn              *QUICConn
	readSecret        map[QUICEncryptionLevel]suiteSecret
	writeSecret       map[QUICEncryptionLevel]suiteSecret
	ticketOpts        QUICSessionTicketOptions
	onResumeSession   func(*SessionState)
	gotParams         []byte
	earlyDataRejected bool
	complete          bool
}

func newTestQUICClient(t *testing.T, config *Config) *testQUICConn {
	q := &testQUICConn{t: t}
	q.conn = QUICClient(&QUICConfig{
		TLSConfig: config,
	})
	t.Cleanup(func() {
		q.conn.Close()
	})
	return q
}

func newTestQUICServer(t *testing.T, config *Config) *testQUICConn {
	q := &testQUICConn{t: t}
	q.conn = QUICServer(&QUICConfig{
		TLSConfig: config,
	})
	t.Cleanup(func() {
		q.conn.Close()
	})
	return q
}

type suiteSecret struct {
	suite  uint16
	secret []byte
}

func (q *testQUICConn) setReadSecret(level QUICEncryptionLevel, suite uint16, secret []byte) {
	if _, ok := q.writeSecret[level]; !ok && level != QUICEncryptionLevelEarly {
		q.t.Errorf("SetReadSecret for level %v called before SetWriteSecret", level)
	}
	if level == QUICEncryptionLevelApplication && !q.complete {
		q.t.Errorf("SetReadSecret for level %v called before HandshakeComplete", level)
	}
	if _, ok := q.readSecret[level]; ok {
		q.t.Errorf("SetReadSecret for level %v called twice", level)
	}
	if q.readSecret == nil {
		q.readSecret = map[QUICEncryptionLevel]suiteSecret{}
	}
	switch level {
	case QUICEncryptionLevelHandshake,
		QUICEncryptionLevelEarly,
		QUICEncryptionLevelApplication:
		q.readSecret[level] = suiteSecret{suite, secret}
	default:
		q.t.Errorf("SetReadSecret for unexpected level %v", level)
	}
}

func (q *testQUICConn) setWriteSecret(level QUICEncryptionLevel, suite uint16, secret []byte) {
	if _, ok := q.writeSecret[level]; ok {
		q.t.Errorf("SetWriteSecret for level %v called twice", level)
	}
	if q.writeSecret == nil {
		q.writeSecret = map[QUICEncryptionLevel]suiteSecret{}
	}
	switch level {
	case QUICEncryptionLevelHandshake,
		QUICEncryptionLevelEarly,
		QUICEncryptionLevelApplication:
		q.writeSecret[level] = suiteSecret{suite, secret}
	default:
		q.t.Errorf("SetWriteSecret for unexpected level %v", level)
	}
}

var errTransportParametersRequired = errors.New("transport parameters required")

func runTestQUICConnection(ctx context.Context, cli, srv *testQUICConn, onEvent func(e QUICEvent, src, dst *testQUICConn) bool) error {
	a, b := cli, srv
	for _, c := range []*testQUICConn{a, b} {
		if !c.conn.conn.quic.started {
			if err := c.conn.Start(ctx); err != nil {
				return err
			}
		}
	}
	idleCount := 0
	for {
		e := a.conn.NextEvent()
		if onEvent != nil && onEvent(e, a, b) {
			continue
		}
		switch e.Kind {
		case QUICNoEvent:
			idleCount++
			if idleCount == 2 {
				if !a.complete || !b.complete {
					return errors.New("handshake incomplete")
				}
				return nil
			}
			a, b = b, a
		case QUICSetReadSecret:
			a.setReadSecret(e.Level, e.Suite, e.Data)
		case QUICSetWriteSecret:
			a.setWriteSecret(e.Level, e.Suite, e.Data)
		case QUICWriteData:
			if err := b.conn.HandleData(e.Level, e.Data); err != nil {
				return err
			}
		case QUICTransportParameters:
			a.gotParams = e.Data
			if a.gotParams == nil {
				a.gotParams = []byte{}
			}
		case QUICTransportParametersRequired:
			return errTransportParametersRequired
		case QUICHandshakeDone:
			a.complete = true
			if a == srv {
				if err := srv.conn.SendSessionTicket(srv.ticketOpts); err != nil {
					return err
				}
			}
		case QUICResumeSession:
			if a.onResumeSession != nil {
				a.onResumeSession(e.SessionState)
			}
		case QUICRejectedEarlyData:
			a.earlyDataRejected = true
		}
		if e.Kind != QUICNoEvent {
			idleCount = 0
		}
	}
}

func TestQUICConnection(t *testing.T) {
	config := testConfig.Clone()
	config.MinVersion = VersionTLS13

	cli := newTestQUICClient(t, config)
	cli.conn.SetTransportParameters(nil)

	srv := newTestQUICServer(t, config)
	srv.conn.SetTransportParameters(nil)

	if err := runTestQUICConnection(context.Background(), cli, srv, nil); err != nil {
		t.Fatalf("error during connection handshake: %v", err)
	}

	if _, ok := cli.readSecret[QUICEncryptionLevelHandshake]; !ok {
		t.Errorf("client has no Handshake secret")
	}
	if _, ok := cli.readSecret[QUICEncryptionLevelApplication]; !ok {
		t.Errorf("client has no Application secret")
	}
	if _, ok := srv.readSecret[QUICEncryptionLevelHandshake]; !ok {
		t.Errorf("server has no Handshake secret")
	}
	if _, ok := srv.readSecret[QUICEncryptionLevelApplication]; !ok {
		t.Errorf("server has no Application secret")
	}
	for _, level := range []QUICEncryptionLevel{QUICEncryptionLevelHandshake, QUICEncryptionLevelApplication} {
		if _, ok := cli.readSecret[level]; !ok {
			t.Errorf("client has no %v read secret", level)
		}
		if _, ok := srv.readSecret[level]; !ok {
			t.Errorf("server has no %v read secret", level)
		}
		if !reflect.DeepEqual(cli.readSecret[level], srv.writeSecret[level]) {
			t.Errorf("client read secret does not match server write secret for level %v", level)
		}
		if !reflect.DeepEqual(cli.writeSecret[level], srv.readSecret[level]) {
			t.Errorf("client write secret does not match server read secret for level %v", level)
		}
	}
}

func TestQUICSessionResumption(t *testing.T) {
	clientConfig := testConfig.Clone()
	clientConfig.MinVersion = VersionTLS13
	clientConfig.ClientSessionCache = NewLRUClientSessionCache(1)
	clientConfig.ServerName = "example.go.dev"

	serverConfig := testConfig.Clone()
	serverConfig.MinVersion = VersionTLS13

	cli := newTestQUICClient(t, clientConfig)
	cli.conn.SetTransportParameters(nil)
	srv := newTestQUICServer(t, serverConfig)
	srv.conn.SetTransportParameters(nil)
	if err := runTestQUICConnection(context.Background(), cli, srv, nil); err != nil {
		t.Fatalf("error during first connection handshake: %v", err)
	}
	if cli.conn.ConnectionState().DidResume {
		t.Errorf("first connection unexpectedly used session resumption")
	}

	cli2 := newTestQUICClient(t, clientConfig)
	cli2.conn.SetTransportParameters(nil)
	srv2 := newTestQUICServer(t, serverConfig)
	srv2.conn.SetTransportParameters(nil)
	if err := runTestQUICConnection(context.Background(), cli2, srv2, nil); err != nil {
		t.Fatalf("error during second connection handshake: %v", err)
	}
	if !cli2.conn.ConnectionState().DidResume {
		t.Errorf("second connection did not use session resumption")
	}
}

func TestQUICFragmentaryData(t *testing.T) {
	clientConfig := testConfig.Clone()
	clientConfig.MinVersion = VersionTLS13
	clientConfig.ClientSessionCache = NewLRUClientSessionCache(1)
	clientConfig.ServerName = "example.go.dev"

	serverConfig := testConfig.Clone()
	serverConfig.MinVersion = VersionTLS13

	cli := newTestQUICClient(t, clientConfig)
	cli.conn.SetTransportParameters(nil)
	srv := newTestQUICServer(t, serverConfig)
	srv.conn.SetTransportParameters(nil)
	onEvent := func(e QUICEvent, src, dst *testQUICConn) bool {
		if e.Kind == QUICWriteData {
			// Provide the data one byte at a time.
			for i := range e.Data {
				if err := dst.conn.HandleData(e.Level, e.Data[i:i+1]); err != nil {
					t.Errorf("HandleData: %v", err)
					break
				}
			}
			return true
		}
		return false
	}
	if err := runTestQUICConnection(context.Background(), cli, srv, onEvent); err != nil {
		t.Fatalf("error during first connection handshake: %v", err)
	}
}

func TestQUICPostHandshakeClientAuthentication(t *testing.T) {
	// RFC 9001, Section 4.4.
	config := testConfig.Clone()
	config.MinVersion = VersionTLS13
	cli := newTestQUICClient(t, config)
	cli.conn.SetTransportParameters(nil)
	srv := newTestQUICServer(t, config)
	srv.conn.SetTransportParameters(nil)
	if err := runTestQUICConnection(context.Background(), cli, srv, nil); err != nil {
		t.Fatalf("error during connection handshake: %v", err)
	}

	certReq := new(certificateRequestMsgTLS13)
	certReq.ocspStapling = true
	certReq.scts = true
	certReq.supportedSignatureAlgorithms = supportedSignatureAlgorithms()
	certReqBytes, err := certReq.marshal()
	if err != nil {
		t.Fatal(err)
	}
	if err := cli.conn.HandleData(QUICEncryptionLevelApplication, append([]byte{
		byte(typeCertificateRequest),
		byte(0), byte(0), byte(len(certReqBytes)),
	}, certReqBytes...)); err == nil {
		t.Fatalf("post-handshake authentication request: got no error, want one")
	}
}

func TestQUICPostHandshakeKeyUpdate(t *testing.T) {
	// RFC 9001, Section 6.
	config := testConfig.Clone()
	config.MinVersion = VersionTLS13
	cli := newTestQUICClient(t, config)
	cli.conn.SetTransportParameters(nil)
	srv := newTestQUICServer(t, config)
	srv.conn.SetTransportParameters(nil)
	if err := runTestQUICConnection(context.Background(), cli, srv, nil); err != nil {
		t.Fatalf("error during connection handshake: %v", err)
	}

	keyUpdate := new(keyUpdateMsg)
	keyUpdateBytes, err := keyUpdate.marshal()
	if err != nil {
		t.Fatal(err)
	}
	if err := cli.conn.HandleData(QUICEncryptionLevelApplication, append([]byte{
		byte(typeKeyUpdate),
		byte(0), byte(0), byte(len(keyUpdateBytes)),
	}, keyUpdateBytes...)); !errors.Is(err, alertUnexpectedMessage) {
		t.Fatalf("key update request: got error %v, want alertUnexpectedMessage", err)
	}
}

func TestQUICPostHandshakeMessageTooLarge(t *testing.T) {
	config := testConfig.Clone()
	config.MinVersion = VersionTLS13
	cli := newTestQUICClient(t, config)
	cli.conn.SetTransportParameters(nil)
	srv := newTestQUICServer(t, config)
	srv.conn.SetTransportParameters(nil)
	if err := runTestQUICConnection(context.Background(), cli, srv, nil); err != nil {
		t.Fatalf("error during connection handshake: %v", err)
	}

	size := maxHandshake + 1
	if err := cli.conn.HandleData(QUICEncryptionLevelApplication, []byte{
		byte(typeNewSessionTicket),
		byte(size >> 16),
		byte(size >> 8),
		byte(size),
	}); err == nil {
		t.Fatalf("%v-byte post-handshake message: got no error, want one", size)
	}
}

func TestQUICHandshakeError(t *testing.T) {
	clientConfig := testConfig.Clone()
	clientConfig.MinVersion = VersionTLS13
	clientConfig.InsecureSkipVerify = false
	clientConfig.ServerName = "name"

	serverConfig := testConfig.Clone()
	serverConfig.MinVersion = VersionTLS13

	cli := newTestQUICClient(t, clientConfig)
	cli.conn.SetTransportParameters(nil)
	srv := newTestQUICServer(t, serverConfig)
	srv.conn.SetTransportParameters(nil)
	err := runTestQUICConnection(context.Background(), cli, srv, nil)
	if !errors.Is(err, AlertError(alertBadCertificate)) {
		t.Errorf("connection handshake terminated with error %q, want alertBadCertificate", err)
	}
	var e *CertificateVerificationError
	if !errors.As(err, &e) {
		t.Errorf("connection handshake terminated with error %q, want CertificateVerificationError", err)
	}
}

// Test that QUICConn.ConnectionState can be used during the handshake,
// and that it reports the application protocol as soon as it has been
// negotiated.
func TestQUICConnectionState(t *testing.T) {
	config := testConfig.Clone()
	config.MinVersion = VersionTLS13
	config.NextProtos = []string{"h3"}
	cli := newTestQUICClient(t, config)
	cli.conn.SetTransportParameters(nil)
	srv := newTestQUICServer(t, config)
	srv.conn.SetTransportParameters(nil)
	onEvent := func(e QUICEvent, src, dst *testQUICConn) bool {
		cliCS := cli.conn.ConnectionState()
		if _, ok := cli.readSecret[QUICEncryptionLevelApplication]; ok {
			if want, got := cliCS.NegotiatedProtocol, "h3"; want != got {
				t.Errorf("cli.ConnectionState().NegotiatedProtocol = %q, want %q", want, got)
			}
		}
		srvCS := srv.conn.ConnectionState()
		if _, ok := srv.readSecret[QUICEncryptionLevelHandshake]; ok {
			if want, got := srvCS.NegotiatedProtocol, "h3"; want != got {
				t.Errorf("srv.ConnectionState().NegotiatedProtocol = %q, want %q", want, got)
			}
		}
		return false
	}
	if err := runTestQUICConnection(context.Background(), cli, srv, onEvent); err != nil {
		t.Fatalf("error during connection handshake: %v", err)
	}
}

func TestQUICStartContextPropagation(t *testing.T) {
	const key = "key"
	const value = "value"
	ctx := context.WithValue(context.Background(), key, value)
	config := testConfig.Clone()
	config.MinVersion = VersionTLS13
	calls := 0
	config.GetConfigForClient = func(info *ClientHelloInfo) (*Config, error) {
		calls++
		got, _ := info.Context().Value(key).(string)
		if got != value {
			t.Errorf("GetConfigForClient context key %q has value %q, want %q", key, got, value)
		}
		return nil, nil
	}
	cli := newTestQUICClient(t, config)
	cli.conn.SetTransportParameters(nil)
	srv := newTestQUICServer(t, config)
	srv.conn.SetTransportParameters(nil)
	if err := runTestQUICConnection(ctx, cli, srv, nil); err != nil {
		t.Fatalf("error during connection handshake: %v", err)
	}
	if calls != 1 {
		t.Errorf("GetConfigForClient called %v times, want 1", calls)
	}
}

func TestQUICDelayedTransportParameters(t *testing.T) {
	clientConfig := testConfig.Clone()
	clientConfig.MinVersion = VersionTLS13
	clientConfig.ClientSessionCache = NewLRUClientSessionCache(1)
	clientConfig.ServerName = "example.go.dev"

	serverConfig := testConfig.Clone()
	serverConfig.MinVersion = VersionTLS13

	cliParams := "client params"
	srvParams := "server params"

	cli := newTestQUICClient(t, clientConfig)
	srv := newTestQUICServer(t, serverConfig)
	if err := runTestQUICConnection(context.Background(), cli, srv, nil); err != errTransportParametersRequired {
		t.Fatalf("handshake with no client parameters: %v; want errTransportParametersRequired", err)
	}
	cli.conn.SetTransportParameters([]byte(cliParams))
	if err := runTestQUICConnection(context.Background(), cli, srv, nil); err != errTransportParametersRequired {
		t.Fatalf("handshake with no server parameters: %v; want errTransportParametersRequired", err)
	}
	srv.conn.SetTransportParameters([]byte(srvParams))
	if err := runTestQUICConnection(context.Background(), cli, srv, nil); err != nil {
		t.Fatalf("error during connection handshake: %v", err)
	}

	if got, want := string(cli.gotParams), srvParams; got != want {
		t.Errorf("client got transport params: %q, want %q", got, want)
	}
	if got, want := string(srv.gotParams), cliParams; got != want {
		t.Errorf("server got transport params: %q, want %q", got, want)
	}
}

func TestQUICEmptyTransportParameters(t *testing.T) {
	config := testConfig.Clone()
	config.MinVersion = VersionTLS13

	cli := newTestQUICClient(t, config)
	cli.conn.SetTransportParameters(nil)
	srv := newTestQUICServer(t, config)
	srv.conn.SetTransportParameters(nil)
	if err := runTestQUICConnection(context.Background(), cli, srv, nil); err != nil {
		t.Fatalf("error during connection handshake: %v", err)
	}

	if cli.gotParams == nil {
		t.Errorf("client did not get transport params")
	}
	if srv.gotParams == nil {
		t.Errorf("server did not get transport params")
	}
	if len(cli.gotParams) != 0 {
		t.Errorf("client got transport params: %v, want empty", cli.gotParams)
	}
	if len(srv.gotParams) != 0 {
		t.Errorf("server got transport params: %v, want empty", srv.gotParams)
	}
}

func TestQUICCanceledWaitingForData(t *testing.T) {
	config := testConfig.Clone()
	config.MinVersion = VersionTLS13
	cli := newTestQUICClient(t, config)
	cli.conn.SetTransportParameters(nil)
	cli.conn.Start(context.Background())
	for cli.conn.NextEvent().Kind != QUICNoEvent {
	}
	err := cli.conn.Close()
	if !errors.Is(err, alertCloseNotify) {
		t.Errorf("conn.Close() = %v, want alertCloseNotify", err)
	}
}

func TestQUICCanceledWaitingForTransportParams(t *testing.T) {
	config := testConfig.Clone()
	config.MinVersion = VersionTLS13
	cli := newTestQUICClient(t, config)
	cli.conn.Start(context.Background())
	for cli.conn.NextEvent().Kind != QUICTransportParametersRequired {
	}
	err := cli.conn.Close()
	if !errors.Is(err, alertCloseNotify) {
		t.Errorf("conn.Close() = %v, want alertCloseNotify", err)
	}
}

func TestQUICEarlyData(t *testing.T) {
	clientConfig := testConfig.Clone()
	clientConfig.MinVersion = VersionTLS13
	clientConfig.ClientSessionCache = NewLRUClientSessionCache(1)
	clientConfig.ServerName = "example.go.dev"
	clientConfig.NextProtos = []string{"h3"}

	serverConfig := testConfig.Clone()
	serverConfig.MinVersion = VersionTLS13
	serverConfig.NextProtos = []string{"h3"}

	cli := newTestQUICClient(t, clientConfig)
	cli.conn.SetTransportParameters(nil)
	srv := newTestQUICServer(t, serverConfig)
	srv.conn.SetTransportParameters(nil)
	srv.ticketOpts.EarlyData = true
	if err := runTestQUICConnection(context.Background(), cli, srv, nil); err != nil {
		t.Fatalf("error during first connection handshake: %v", err)
	}
	if cli.conn.ConnectionState().DidResume {
		t.Errorf("first connection unexpectedly used session resumption")
	}

	cli2 := newTestQUICClient(t, clientConfig)
	cli2.conn.SetTransportParameters(nil)
	srv2 := newTestQUICServer(t, serverConfig)
	srv2.conn.SetTransportParameters(nil)
	if err := runTestQUICConnection(context.Background(), cli2, srv2, nil); err != nil {
		t.Fatalf("error during second connection handshake: %v", err)
	}
	if !cli2.conn.ConnectionState().DidResume {
		t.Errorf("second connection did not use session resumption")
	}
	cliSecret := cli2.writeSecret[QUICEncryptionLevelEarly]
	if cliSecret.secret == nil {
		t.Errorf("client did not receive early data write secret")
	}
	srvSecret := srv2.readSecret[QUICEncryptionLevelEarly]
	if srvSecret.secret == nil {
		t.Errorf("server did not receive early data read secret")
	}
	if cliSecret.suite != srvSecret.suite || !bytes.Equal(cliSecret.secret, srvSecret.secret) {
		t.Errorf("client early data secret does not match server")
	}
}

func TestQUICEarlyDataDeclined(t *testing.T) {
	t.Run("server", func(t *testing.T) {
		testQUICEarlyDataDeclined(t, true)
	})
	t.Run("client", func(t *testing.T) {
		testQUICEarlyDataDeclined(t, false)
	})
}

func testQUICEarlyDataDeclined(t *testing.T, server bool) {
	clientConfig := testConfig.Clone()
	clientConfig.MinVersion = VersionTLS13
	clientConfig.ClientSessionCache = NewLRUClientSessionCache(1)
	clientConfig.ServerName = "example.go.dev"
	clientConfig.NextProtos = []string{"h3"}

	serverConfig := testConfig.Clone()
	serverConfig.MinVersion = VersionTLS13
	serverConfig.NextProtos = []string{"h3"}

	cli := newTestQUICClient(t, clientConfig)
	cli.conn.SetTransportParameters(nil)
	srv := newTestQUICServer(t, serverConfig)
	srv.conn.SetTransportParameters(nil)
	srv.ticketOpts.EarlyData = true
	if err := runTestQUICConnection(context.Background(), cli, srv, nil); err != nil {
		t.Fatalf("error during first connection handshake: %v", err)
	}
	if cli.conn.ConnectionState().DidResume {
		t.Errorf("first connection unexpectedly used session resumption")
	}

	cli2 := newTestQUICClient(t, clientConfig)
	cli2.conn.SetTransportParameters(nil)
	srv2 := newTestQUICServer(t, serverConfig)
	srv2.conn.SetTransportParameters(nil)
	declineEarlyData := func(state *SessionState) {
		state.EarlyData = false
	}
	if server {
		srv2.onResumeSession = declineEarlyData
	} else {
		cli2.onResumeSession = declineEarlyData
	}
	if err := runTestQUICConnection(context.Background(), cli2, srv2, nil); err != nil {
		t.Fatalf("error during second connection handshake: %v", err)
	}
	if !cli2.conn.ConnectionState().DidResume {
		t.Errorf("second connection did not use session resumption")
	}
	_, cliEarlyData := cli2.writeSecret[QUICEncryptionLevelEarly]
	if server {
		if !cliEarlyData {
			t.Errorf("client did not receive early data write secret")
		}
		if !cli2.earlyDataRejected {
			t.Errorf("client did not receive QUICEarlyDataRejected")
		}
	}
	if _, srvEarlyData := srv2.readSecret[QUICEncryptionLevelEarly]; srvEarlyData {
		t.Errorf("server received early data read secret")
	}
}
