// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bytes"
	"math/rand"
	"reflect"
	"strings"
	"testing"
	"testing/quick"
	"time"
)

var tests = []any{
	&clientHelloMsg{},
	&serverHelloMsg{},
	&finishedMsg{},

	&certificateMsg{},
	&certificateRequestMsg{},
	&certificateVerifyMsg{
		hasSignatureAlgorithm: true,
	},
	&certificateStatusMsg{},
	&clientKeyExchangeMsg{},
	&newSessionTicketMsg{},
	&sessionState{},
	&sessionStateTLS13{},
	&encryptedExtensionsMsg{},
	&endOfEarlyDataMsg{},
	&keyUpdateMsg{},
	&newSessionTicketMsgTLS13{},
	&certificateRequestMsgTLS13{},
	&certificateMsgTLS13{},
}

func TestMarshalUnmarshal(t *testing.T) {
	rand := rand.New(rand.NewSource(time.Now().UnixNano()))

	for i, iface := range tests {
		ty := reflect.ValueOf(iface).Type()

		n := 100
		if testing.Short() {
			n = 5
		}
		for j := 0; j < n; j++ {
			v, ok := quick.Value(ty, rand)
			if !ok {
				t.Errorf("#%d: failed to create value", i)
				break
			}

			m1 := v.Interface().(handshakeMessage)
			marshaled := m1.marshal()
			m2 := iface.(handshakeMessage)
			if !m2.unmarshal(marshaled) {
				t.Errorf("#%d failed to unmarshal %#v %x", i, m1, marshaled)
				break
			}
			m2.marshal() // to fill any marshal cache in the message

			if !reflect.DeepEqual(m1, m2) {
				t.Errorf("#%d got:%#v want:%#v %x", i, m2, m1, marshaled)
				break
			}

			if i >= 3 {
				// The first three message types (ClientHello,
				// ServerHello and Finished) are allowed to
				// have parsable prefixes because the extension
				// data is optional and the length of the
				// Finished varies across versions.
				for j := 0; j < len(marshaled); j++ {
					if m2.unmarshal(marshaled[0:j]) {
						t.Errorf("#%d unmarshaled a prefix of length %d of %#v", i, j, m1)
						break
					}
				}
			}
		}
	}
}

func TestFuzz(t *testing.T) {
	rand := rand.New(rand.NewSource(0))
	for _, iface := range tests {
		m := iface.(handshakeMessage)

		for j := 0; j < 1000; j++ {
			len := rand.Intn(100)
			bytes := randomBytes(len, rand)
			// This just looks for crashes due to bounds errors etc.
			m.unmarshal(bytes)
		}
	}
}

func randomBytes(n int, rand *rand.Rand) []byte {
	r := make([]byte, n)
	if _, err := rand.Read(r); err != nil {
		panic("rand.Read failed: " + err.Error())
	}
	return r
}

func randomString(n int, rand *rand.Rand) string {
	b := randomBytes(n, rand)
	return string(b)
}

func (*clientHelloMsg) Generate(rand *rand.Rand, size int) reflect.Value {
	m := &clientHelloMsg{}
	m.vers = uint16(rand.Intn(65536))
	m.random = randomBytes(32, rand)
	m.sessionId = randomBytes(rand.Intn(32), rand)
	m.cipherSuites = make([]uint16, rand.Intn(63)+1)
	for i := 0; i < len(m.cipherSuites); i++ {
		cs := uint16(rand.Int31())
		if cs == scsvRenegotiation {
			cs += 1
		}
		m.cipherSuites[i] = cs
	}
	m.compressionMethods = randomBytes(rand.Intn(63)+1, rand)
	if rand.Intn(10) > 5 {
		m.serverName = randomString(rand.Intn(255), rand)
		for strings.HasSuffix(m.serverName, ".") {
			m.serverName = m.serverName[:len(m.serverName)-1]
		}
	}
	m.ocspStapling = rand.Intn(10) > 5
	m.supportedPoints = randomBytes(rand.Intn(5)+1, rand)
	m.supportedCurves = make([]CurveID, rand.Intn(5)+1)
	for i := range m.supportedCurves {
		m.supportedCurves[i] = CurveID(rand.Intn(30000) + 1)
	}
	if rand.Intn(10) > 5 {
		m.ticketSupported = true
		if rand.Intn(10) > 5 {
			m.sessionTicket = randomBytes(rand.Intn(300), rand)
		} else {
			m.sessionTicket = make([]byte, 0)
		}
	}
	if rand.Intn(10) > 5 {
		m.supportedSignatureAlgorithms = supportedSignatureAlgorithms
	}
	if rand.Intn(10) > 5 {
		m.supportedSignatureAlgorithmsCert = supportedSignatureAlgorithms
	}
	for i := 0; i < rand.Intn(5); i++ {
		m.alpnProtocols = append(m.alpnProtocols, randomString(rand.Intn(20)+1, rand))
	}
	if rand.Intn(10) > 5 {
		m.scts = true
	}
	if rand.Intn(10) > 5 {
		m.secureRenegotiationSupported = true
		m.secureRenegotiation = randomBytes(rand.Intn(50)+1, rand)
	}
	for i := 0; i < rand.Intn(5); i++ {
		m.supportedVersions = append(m.supportedVersions, uint16(rand.Intn(0xffff)+1))
	}
	if rand.Intn(10) > 5 {
		m.cookie = randomBytes(rand.Intn(500)+1, rand)
	}
	for i := 0; i < rand.Intn(5); i++ {
		var ks keyShare
		ks.group = CurveID(rand.Intn(30000) + 1)
		ks.data = randomBytes(rand.Intn(200)+1, rand)
		m.keyShares = append(m.keyShares, ks)
	}
	switch rand.Intn(3) {
	case 1:
		m.pskModes = []uint8{pskModeDHE}
	case 2:
		m.pskModes = []uint8{pskModeDHE, pskModePlain}
	}
	for i := 0; i < rand.Intn(5); i++ {
		var psk pskIdentity
		psk.obfuscatedTicketAge = uint32(rand.Intn(500000))
		psk.label = randomBytes(rand.Intn(500)+1, rand)
		m.pskIdentities = append(m.pskIdentities, psk)
		m.pskBinders = append(m.pskBinders, randomBytes(rand.Intn(50)+32, rand))
	}
	if rand.Intn(10) > 5 {
		m.earlyData = true
	}

	return reflect.ValueOf(m)
}

func (*serverHelloMsg) Generate(rand *rand.Rand, size int) reflect.Value {
	m := &serverHelloMsg{}
	m.vers = uint16(rand.Intn(65536))
	m.random = randomBytes(32, rand)
	m.sessionId = randomBytes(rand.Intn(32), rand)
	m.cipherSuite = uint16(rand.Int31())
	m.compressionMethod = uint8(rand.Intn(256))
	m.supportedPoints = randomBytes(rand.Intn(5)+1, rand)

	if rand.Intn(10) > 5 {
		m.ocspStapling = true
	}
	if rand.Intn(10) > 5 {
		m.ticketSupported = true
	}
	if rand.Intn(10) > 5 {
		m.alpnProtocol = randomString(rand.Intn(32)+1, rand)
	}

	for i := 0; i < rand.Intn(4); i++ {
		m.scts = append(m.scts, randomBytes(rand.Intn(500)+1, rand))
	}

	if rand.Intn(10) > 5 {
		m.secureRenegotiationSupported = true
		m.secureRenegotiation = randomBytes(rand.Intn(50)+1, rand)
	}
	if rand.Intn(10) > 5 {
		m.supportedVersion = uint16(rand.Intn(0xffff) + 1)
	}
	if rand.Intn(10) > 5 {
		m.cookie = randomBytes(rand.Intn(500)+1, rand)
	}
	if rand.Intn(10) > 5 {
		for i := 0; i < rand.Intn(5); i++ {
			m.serverShare.group = CurveID(rand.Intn(30000) + 1)
			m.serverShare.data = randomBytes(rand.Intn(200)+1, rand)
		}
	} else if rand.Intn(10) > 5 {
		m.selectedGroup = CurveID(rand.Intn(30000) + 1)
	}
	if rand.Intn(10) > 5 {
		m.selectedIdentityPresent = true
		m.selectedIdentity = uint16(rand.Intn(0xffff))
	}

	return reflect.ValueOf(m)
}

func (*encryptedExtensionsMsg) Generate(rand *rand.Rand, size int) reflect.Value {
	m := &encryptedExtensionsMsg{}

	if rand.Intn(10) > 5 {
		m.alpnProtocol = randomString(rand.Intn(32)+1, rand)
	}

	return reflect.ValueOf(m)
}

func (*certificateMsg) Generate(rand *rand.Rand, size int) reflect.Value {
	m := &certificateMsg{}
	numCerts := rand.Intn(20)
	m.certificates = make([][]byte, numCerts)
	for i := 0; i < numCerts; i++ {
		m.certificates[i] = randomBytes(rand.Intn(10)+1, rand)
	}
	return reflect.ValueOf(m)
}

func (*certificateRequestMsg) Generate(rand *rand.Rand, size int) reflect.Value {
	m := &certificateRequestMsg{}
	m.certificateTypes = randomBytes(rand.Intn(5)+1, rand)
	for i := 0; i < rand.Intn(100); i++ {
		m.certificateAuthorities = append(m.certificateAuthorities, randomBytes(rand.Intn(15)+1, rand))
	}
	return reflect.ValueOf(m)
}

func (*certificateVerifyMsg) Generate(rand *rand.Rand, size int) reflect.Value {
	m := &certificateVerifyMsg{}
	m.hasSignatureAlgorithm = true
	m.signatureAlgorithm = SignatureScheme(rand.Intn(30000))
	m.signature = randomBytes(rand.Intn(15)+1, rand)
	return reflect.ValueOf(m)
}

func (*certificateStatusMsg) Generate(rand *rand.Rand, size int) reflect.Value {
	m := &certificateStatusMsg{}
	m.response = randomBytes(rand.Intn(10)+1, rand)
	return reflect.ValueOf(m)
}

func (*clientKeyExchangeMsg) Generate(rand *rand.Rand, size int) reflect.Value {
	m := &clientKeyExchangeMsg{}
	m.ciphertext = randomBytes(rand.Intn(1000)+1, rand)
	return reflect.ValueOf(m)
}

func (*finishedMsg) Generate(rand *rand.Rand, size int) reflect.Value {
	m := &finishedMsg{}
	m.verifyData = randomBytes(12, rand)
	return reflect.ValueOf(m)
}

func (*newSessionTicketMsg) Generate(rand *rand.Rand, size int) reflect.Value {
	m := &newSessionTicketMsg{}
	m.ticket = randomBytes(rand.Intn(4), rand)
	return reflect.ValueOf(m)
}

func (*sessionState) Generate(rand *rand.Rand, size int) reflect.Value {
	s := &sessionState{}
	s.vers = uint16(rand.Intn(10000))
	s.cipherSuite = uint16(rand.Intn(10000))
	s.masterSecret = randomBytes(rand.Intn(100)+1, rand)
	s.createdAt = uint64(rand.Int63())
	for i := 0; i < rand.Intn(20); i++ {
		s.certificates = append(s.certificates, randomBytes(rand.Intn(500)+1, rand))
	}
	return reflect.ValueOf(s)
}

func (*sessionStateTLS13) Generate(rand *rand.Rand, size int) reflect.Value {
	s := &sessionStateTLS13{}
	s.cipherSuite = uint16(rand.Intn(10000))
	s.resumptionSecret = randomBytes(rand.Intn(100)+1, rand)
	s.createdAt = uint64(rand.Int63())
	for i := 0; i < rand.Intn(2)+1; i++ {
		s.certificate.Certificate = append(
			s.certificate.Certificate, randomBytes(rand.Intn(500)+1, rand))
	}
	if rand.Intn(10) > 5 {
		s.certificate.OCSPStaple = randomBytes(rand.Intn(100)+1, rand)
	}
	if rand.Intn(10) > 5 {
		for i := 0; i < rand.Intn(2)+1; i++ {
			s.certificate.SignedCertificateTimestamps = append(
				s.certificate.SignedCertificateTimestamps, randomBytes(rand.Intn(500)+1, rand))
		}
	}
	return reflect.ValueOf(s)
}

func (*endOfEarlyDataMsg) Generate(rand *rand.Rand, size int) reflect.Value {
	m := &endOfEarlyDataMsg{}
	return reflect.ValueOf(m)
}

func (*keyUpdateMsg) Generate(rand *rand.Rand, size int) reflect.Value {
	m := &keyUpdateMsg{}
	m.updateRequested = rand.Intn(10) > 5
	return reflect.ValueOf(m)
}

func (*newSessionTicketMsgTLS13) Generate(rand *rand.Rand, size int) reflect.Value {
	m := &newSessionTicketMsgTLS13{}
	m.lifetime = uint32(rand.Intn(500000))
	m.ageAdd = uint32(rand.Intn(500000))
	m.nonce = randomBytes(rand.Intn(100), rand)
	m.label = randomBytes(rand.Intn(1000), rand)
	if rand.Intn(10) > 5 {
		m.maxEarlyData = uint32(rand.Intn(500000))
	}
	return reflect.ValueOf(m)
}

func (*certificateRequestMsgTLS13) Generate(rand *rand.Rand, size int) reflect.Value {
	m := &certificateRequestMsgTLS13{}
	if rand.Intn(10) > 5 {
		m.ocspStapling = true
	}
	if rand.Intn(10) > 5 {
		m.scts = true
	}
	if rand.Intn(10) > 5 {
		m.supportedSignatureAlgorithms = supportedSignatureAlgorithms
	}
	if rand.Intn(10) > 5 {
		m.supportedSignatureAlgorithmsCert = supportedSignatureAlgorithms
	}
	if rand.Intn(10) > 5 {
		m.certificateAuthorities = make([][]byte, 3)
		for i := 0; i < 3; i++ {
			m.certificateAuthorities[i] = randomBytes(rand.Intn(10)+1, rand)
		}
	}
	return reflect.ValueOf(m)
}

func (*certificateMsgTLS13) Generate(rand *rand.Rand, size int) reflect.Value {
	m := &certificateMsgTLS13{}
	for i := 0; i < rand.Intn(2)+1; i++ {
		m.certificate.Certificate = append(
			m.certificate.Certificate, randomBytes(rand.Intn(500)+1, rand))
	}
	if rand.Intn(10) > 5 {
		m.ocspStapling = true
		m.certificate.OCSPStaple = randomBytes(rand.Intn(100)+1, rand)
	}
	if rand.Intn(10) > 5 {
		m.scts = true
		for i := 0; i < rand.Intn(2)+1; i++ {
			m.certificate.SignedCertificateTimestamps = append(
				m.certificate.SignedCertificateTimestamps, randomBytes(rand.Intn(500)+1, rand))
		}
	}
	return reflect.ValueOf(m)
}

func TestRejectEmptySCTList(t *testing.T) {
	// RFC 6962, Section 3.3.1 specifies that empty SCT lists are invalid.

	var random [32]byte
	sct := []byte{0x42, 0x42, 0x42, 0x42}
	serverHello := serverHelloMsg{
		vers:   VersionTLS12,
		random: random[:],
		scts:   [][]byte{sct},
	}
	serverHelloBytes := serverHello.marshal()

	var serverHelloCopy serverHelloMsg
	if !serverHelloCopy.unmarshal(serverHelloBytes) {
		t.Fatal("Failed to unmarshal initial message")
	}

	// Change serverHelloBytes so that the SCT list is empty
	i := bytes.Index(serverHelloBytes, sct)
	if i < 0 {
		t.Fatal("Cannot find SCT in ServerHello")
	}

	var serverHelloEmptySCT []byte
	serverHelloEmptySCT = append(serverHelloEmptySCT, serverHelloBytes[:i-6]...)
	// Append the extension length and SCT list length for an empty list.
	serverHelloEmptySCT = append(serverHelloEmptySCT, []byte{0, 2, 0, 0}...)
	serverHelloEmptySCT = append(serverHelloEmptySCT, serverHelloBytes[i+4:]...)

	// Update the handshake message length.
	serverHelloEmptySCT[1] = byte((len(serverHelloEmptySCT) - 4) >> 16)
	serverHelloEmptySCT[2] = byte((len(serverHelloEmptySCT) - 4) >> 8)
	serverHelloEmptySCT[3] = byte(len(serverHelloEmptySCT) - 4)

	// Update the extensions length
	serverHelloEmptySCT[42] = byte((len(serverHelloEmptySCT) - 44) >> 8)
	serverHelloEmptySCT[43] = byte((len(serverHelloEmptySCT) - 44))

	if serverHelloCopy.unmarshal(serverHelloEmptySCT) {
		t.Fatal("Unmarshaled ServerHello with empty SCT list")
	}
}

func TestRejectEmptySCT(t *testing.T) {
	// Not only must the SCT list be non-empty, but the SCT elements must
	// not be zero length.

	var random [32]byte
	serverHello := serverHelloMsg{
		vers:   VersionTLS12,
		random: random[:],
		scts:   [][]byte{nil},
	}
	serverHelloBytes := serverHello.marshal()

	var serverHelloCopy serverHelloMsg
	if serverHelloCopy.unmarshal(serverHelloBytes) {
		t.Fatal("Unmarshaled ServerHello with zero-length SCT")
	}
}
