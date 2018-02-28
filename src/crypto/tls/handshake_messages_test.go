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
)

var tests = []interface{}{
	&clientHelloMsg{},
	&serverHelloMsg{},
	&finishedMsg{},

	&certificateMsg{},
	&certificateRequestMsg{},
	&certificateVerifyMsg{},
	&certificateStatusMsg{},
	&clientKeyExchangeMsg{},
	&nextProtoMsg{},
	&newSessionTicketMsg{},
	&sessionState{},
}

type testMessage interface {
	marshal() []byte
	unmarshal([]byte) bool
	equal(interface{}) bool
}

func TestMarshalUnmarshal(t *testing.T) {
	rand := rand.New(rand.NewSource(0))

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

			m1 := v.Interface().(testMessage)
			marshaled := m1.marshal()
			m2 := iface.(testMessage)
			if !m2.unmarshal(marshaled) {
				t.Errorf("#%d failed to unmarshal %#v %x", i, m1, marshaled)
				break
			}
			m2.marshal() // to fill any marshal cache in the message

			if !m1.equal(m2) {
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
		m := iface.(testMessage)

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
		m.nextProtoNeg = true
	}
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
		m.supportedCurves[i] = CurveID(rand.Intn(30000))
	}
	if rand.Intn(10) > 5 {
		m.ticketSupported = true
		if rand.Intn(10) > 5 {
			m.sessionTicket = randomBytes(rand.Intn(300), rand)
		}
	}
	if rand.Intn(10) > 5 {
		m.supportedSignatureAlgorithms = supportedSignatureAlgorithms
	}
	m.alpnProtocols = make([]string, rand.Intn(5))
	for i := range m.alpnProtocols {
		m.alpnProtocols[i] = randomString(rand.Intn(20)+1, rand)
	}
	if rand.Intn(10) > 5 {
		m.scts = true
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

	if rand.Intn(10) > 5 {
		m.nextProtoNeg = true

		n := rand.Intn(10)
		m.nextProtos = make([]string, n)
		for i := 0; i < n; i++ {
			m.nextProtos[i] = randomString(20, rand)
		}
	}

	if rand.Intn(10) > 5 {
		m.ocspStapling = true
	}
	if rand.Intn(10) > 5 {
		m.ticketSupported = true
	}
	m.alpnProtocol = randomString(rand.Intn(32)+1, rand)

	if rand.Intn(10) > 5 {
		numSCTs := rand.Intn(4)
		m.scts = make([][]byte, numSCTs)
		for i := range m.scts {
			m.scts[i] = randomBytes(rand.Intn(500), rand)
		}
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
	numCAs := rand.Intn(100)
	m.certificateAuthorities = make([][]byte, numCAs)
	for i := 0; i < numCAs; i++ {
		m.certificateAuthorities[i] = randomBytes(rand.Intn(15)+1, rand)
	}
	return reflect.ValueOf(m)
}

func (*certificateVerifyMsg) Generate(rand *rand.Rand, size int) reflect.Value {
	m := &certificateVerifyMsg{}
	m.signature = randomBytes(rand.Intn(15)+1, rand)
	return reflect.ValueOf(m)
}

func (*certificateStatusMsg) Generate(rand *rand.Rand, size int) reflect.Value {
	m := &certificateStatusMsg{}
	if rand.Intn(10) > 5 {
		m.statusType = statusTypeOCSP
		m.response = randomBytes(rand.Intn(10)+1, rand)
	} else {
		m.statusType = 42
	}
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

func (*nextProtoMsg) Generate(rand *rand.Rand, size int) reflect.Value {
	m := &nextProtoMsg{}
	m.proto = randomString(rand.Intn(255), rand)
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
	s.masterSecret = randomBytes(rand.Intn(100), rand)
	numCerts := rand.Intn(20)
	s.certificates = make([][]byte, numCerts)
	for i := 0; i < numCerts; i++ {
		s.certificates[i] = randomBytes(rand.Intn(10)+1, rand)
	}
	return reflect.ValueOf(s)
}

func TestRejectEmptySCTList(t *testing.T) {
	// https://tools.ietf.org/html/rfc6962#section-3.3.1 specifies that
	// empty SCT lists are invalid.

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
