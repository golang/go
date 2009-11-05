// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bytes";
	"big";
	"crypto/rsa";
	"os";
	"testing";
	"testing/script";
)

type zeroSource struct{}

func (zeroSource) Read(b []byte) (n int, err os.Error) {
	for i := range b {
		b[i] = 0;
	}

	return len(b), nil;
}

var testConfig *Config

func init() {
	testConfig = new(Config);
	testConfig.Time = func() int64 { return 0 };
	testConfig.Rand = zeroSource{};
	testConfig.Certificates = make([]Certificate, 1);
	testConfig.Certificates[0].Certificate = [][]byte{testCertificate};
	testConfig.Certificates[0].PrivateKey = testPrivateKey;
}

func setupServerHandshake() (writeChan chan interface{}, controlChan chan interface{}, msgChan chan interface{}) {
	sh := new(serverHandshake);
	writeChan = make(chan interface{});
	controlChan = make(chan interface{});
	msgChan = make(chan interface{});

	go sh.loop(writeChan, controlChan, msgChan, testConfig);
	return;
}

func testClientHelloFailure(t *testing.T, clientHello interface{}, expectedAlert alertType) {
	writeChan, controlChan, msgChan := setupServerHandshake();
	defer close(msgChan);

	send := script.NewEvent("send", nil, script.Send{msgChan, clientHello});
	recvAlert := script.NewEvent("recv alert", []*script.Event{send}, script.Recv{writeChan, alert{alertLevelError, expectedAlert}});
	close1 := script.NewEvent("msgChan close", []*script.Event{recvAlert}, script.Closed{writeChan});
	recvState := script.NewEvent("recv state", []*script.Event{send}, script.Recv{controlChan, ConnectionState{false, "", expectedAlert}});
	close2 := script.NewEvent("controlChan close", []*script.Event{recvState}, script.Closed{controlChan});

	err := script.Perform(0, []*script.Event{send, recvAlert, close1, recvState, close2});
	if err != nil {
		t.Errorf("Got error: %s", err);
	}
}

func TestSimpleError(t *testing.T) {
	testClientHelloFailure(t, &serverHelloDoneMsg{}, alertUnexpectedMessage);
}

var badProtocolVersions = []uint8{0, 0, 0, 5, 1, 0, 1, 5, 2, 0, 2, 5, 3, 0}

func TestRejectBadProtocolVersion(t *testing.T) {
	clientHello := new(clientHelloMsg);

	for i := 0; i < len(badProtocolVersions); i += 2 {
		clientHello.major = badProtocolVersions[i];
		clientHello.minor = badProtocolVersions[i+1];

		testClientHelloFailure(t, clientHello, alertProtocolVersion);
	}
}

func TestNoSuiteOverlap(t *testing.T) {
	clientHello := &clientHelloMsg{nil, 3, 1, nil, nil, []uint16{0xff00}, []uint8{0}};
	testClientHelloFailure(t, clientHello, alertHandshakeFailure);

}

func TestNoCompressionOverlap(t *testing.T) {
	clientHello := &clientHelloMsg{nil, 3, 1, nil, nil, []uint16{TLS_RSA_WITH_RC4_128_SHA}, []uint8{0xff}};
	testClientHelloFailure(t, clientHello, alertHandshakeFailure);
}

func matchServerHello(v interface{}) bool {
	serverHello, ok := v.(*serverHelloMsg);
	if !ok {
		return false;
	}
	return serverHello.major == 3 &&
		serverHello.minor == 2 &&
		serverHello.cipherSuite == TLS_RSA_WITH_RC4_128_SHA &&
		serverHello.compressionMethod == compressionNone;
}

func TestAlertForwarding(t *testing.T) {
	writeChan, controlChan, msgChan := setupServerHandshake();
	defer close(msgChan);

	a := alert{alertLevelError, alertNoRenegotiation};
	sendAlert := script.NewEvent("send alert", nil, script.Send{msgChan, a});
	recvAlert := script.NewEvent("recv alert", []*script.Event{sendAlert}, script.Recv{writeChan, a});
	closeWriter := script.NewEvent("close writer", []*script.Event{recvAlert}, script.Closed{writeChan});
	closeControl := script.NewEvent("close control", []*script.Event{recvAlert}, script.Closed{controlChan});

	err := script.Perform(0, []*script.Event{sendAlert, recvAlert, closeWriter, closeControl});
	if err != nil {
		t.Errorf("Got error: %s", err);
	}
}

func TestClose(t *testing.T) {
	writeChan, controlChan, msgChan := setupServerHandshake();

	close := script.NewEvent("close", nil, script.Close{msgChan});
	closed1 := script.NewEvent("closed1", []*script.Event{close}, script.Closed{writeChan});
	closed2 := script.NewEvent("closed2", []*script.Event{close}, script.Closed{controlChan});

	err := script.Perform(0, []*script.Event{close, closed1, closed2});
	if err != nil {
		t.Errorf("Got error: %s", err);
	}
}

func matchCertificate(v interface{}) bool {
	cert, ok := v.(*certificateMsg);
	if !ok {
		return false;
	}
	return len(cert.certificates) == 1 &&
		bytes.Compare(cert.certificates[0], testCertificate) == 0;
}

func matchSetCipher(v interface{}) bool {
	_, ok := v.(writerChangeCipherSpec);
	return ok;
}

func matchDone(v interface{}) bool {
	_, ok := v.(*serverHelloDoneMsg);
	return ok;
}

func matchFinished(v interface{}) bool {
	finished, ok := v.(*finishedMsg);
	if !ok {
		return false;
	}
	return bytes.Compare(finished.verifyData, fromHex("29122ae11453e631487b02ed")) == 0;
}

func matchNewCipherSpec(v interface{}) bool {
	_, ok := v.(*newCipherSpec);
	return ok;
}

func TestFullHandshake(t *testing.T) {
	writeChan, controlChan, msgChan := setupServerHandshake();
	defer close(msgChan);

	// The values for this test were obtained from running `gnutls-cli --insecure --debug 9`
	clientHello := &clientHelloMsg{fromHex("0100007603024aef7d77e4686d5dfd9d953dfe280788759ffd440867d687670216da45516b310000340033004500390088001600320044003800870013006600900091008f008e002f004100350084000a00050004008c008d008b008a01000019000900030200010000000e000c0000093132372e302e302e31"), 3, 2, fromHex("4aef7d77e4686d5dfd9d953dfe280788759ffd440867d687670216da45516b31"), nil, []uint16{0x33, 0x45, 0x39, 0x88, 0x16, 0x32, 0x44, 0x38, 0x87, 0x13, 0x66, 0x90, 0x91, 0x8f, 0x8e, 0x2f, 0x41, 0x35, 0x84, 0xa, 0x5, 0x4, 0x8c, 0x8d, 0x8b, 0x8a}, []uint8{0x0}};

	sendHello := script.NewEvent("send hello", nil, script.Send{msgChan, clientHello});
	setVersion := script.NewEvent("set version", []*script.Event{sendHello}, script.Recv{writeChan, writerSetVersion{3, 2}});
	recvHello := script.NewEvent("recv hello", []*script.Event{setVersion}, script.RecvMatch{writeChan, matchServerHello});
	recvCert := script.NewEvent("recv cert", []*script.Event{recvHello}, script.RecvMatch{writeChan, matchCertificate});
	recvDone := script.NewEvent("recv done", []*script.Event{recvCert}, script.RecvMatch{writeChan, matchDone});

	ckx := &clientKeyExchangeMsg{nil, fromHex("872e1fee5f37dd86f3215938ac8de20b302b90074e9fb93097e6b7d1286d0f45abf2daf179deb618bb3c70ed0afee6ee24476ee4649e5a23358143c0f1d9c251")};
	sendCKX := script.NewEvent("send ckx", []*script.Event{recvDone}, script.Send{msgChan, ckx});

	sendCCS := script.NewEvent("send ccs", []*script.Event{sendCKX}, script.Send{msgChan, changeCipherSpec{}});
	recvNCS := script.NewEvent("recv done", []*script.Event{sendCCS}, script.RecvMatch{controlChan, matchNewCipherSpec});

	finished := &finishedMsg{nil, fromHex("c8faca5d242f4423325c5b1a")};
	sendFinished := script.NewEvent("send finished", []*script.Event{recvNCS}, script.Send{msgChan, finished});
	recvFinished := script.NewEvent("recv finished", []*script.Event{sendFinished}, script.RecvMatch{writeChan, matchFinished});
	setCipher := script.NewEvent("set cipher", []*script.Event{sendFinished}, script.RecvMatch{writeChan, matchSetCipher});
	recvConnectionState := script.NewEvent("recv state", []*script.Event{sendFinished}, script.Recv{controlChan, ConnectionState{true, "TLS_RSA_WITH_RC4_128_SHA", 0}});

	err := script.Perform(0, []*script.Event{sendHello, setVersion, recvHello, recvCert, recvDone, sendCKX, sendCCS, recvNCS, sendFinished, setCipher, recvConnectionState, recvFinished});
	if err != nil {
		t.Errorf("Got error: %s", err);
	}
}

var testCertificate = fromHex("3082025930820203a003020102020900c2ec326b95228959300d06092a864886f70d01010505003054310b3009060355040613024155311330110603550408130a536f6d652d53746174653121301f060355040a1318496e7465726e6574205769646769747320507479204c7464310d300b0603550403130474657374301e170d3039313032303232323434355a170d3130313032303232323434355a3054310b3009060355040613024155311330110603550408130a536f6d652d53746174653121301f060355040a1318496e7465726e6574205769646769747320507479204c7464310d300b0603550403130474657374305c300d06092a864886f70d0101010500034b003048024100b2990f49c47dfa8cd400ae6a4d1b8a3b6a13642b23f28b003bfb97790ade9a4cc82b8b2a81747ddec08b6296e53a08c331687ef25c4bf4936ba1c0e6041e9d150203010001a381b73081b4301d0603551d0e0416041478a06086837c9293a8c9b70c0bdabdb9d77eeedf3081840603551d23047d307b801478a06086837c9293a8c9b70c0bdabdb9d77eeedfa158a4563054310b3009060355040613024155311330110603550408130a536f6d652d53746174653121301f060355040a1318496e7465726e6574205769646769747320507479204c7464310d300b0603550403130474657374820900c2ec326b95228959300c0603551d13040530030101ff300d06092a864886f70d0101050500034100ac23761ae1349d85a439caad4d0b932b09ea96de1917c3e0507c446f4838cb3076fb4d431db8c1987e96f1d7a8a2054dea3a64ec99a3f0eda4d47a163bf1f6ac")

func bigFromString(s string) *big.Int {
	ret := new(big.Int);
	ret.SetString(s, 10);
	return ret;
}

var testPrivateKey = &rsa.PrivateKey{
	PublicKey: rsa.PublicKey{
		N: bigFromString("9353930466774385905609975137998169297361893554149986716853295022578535724979677252958524466350471210367835187480748268864277464700638583474144061408845077"),
		E: 65537,
	},
	D: bigFromString("7266398431328116344057699379749222532279343923819063639497049039389899328538543087657733766554155839834519529439851673014800261285757759040931985506583861"),
	P: bigFromString("98920366548084643601728869055592650835572950932266967461790948584315647051443"),
	Q: bigFromString("94560208308847015747498523884063394671606671904944666360068158221458669711639"),
}
