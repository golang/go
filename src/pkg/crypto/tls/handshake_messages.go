// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

type clientHelloMsg struct {
	raw			[]byte;
	major, minor		uint8;
	random			[]byte;
	sessionId		[]byte;
	cipherSuites		[]uint16;
	compressionMethods	[]uint8;
}

func (m *clientHelloMsg) marshal() []byte {
	if m.raw != nil {
		return m.raw
	}

	length := 2 + 32 + 1 + len(m.sessionId) + 2 + len(m.cipherSuites)*2 + 1 + len(m.compressionMethods);
	x := make([]byte, 4+length);
	x[0] = typeClientHello;
	x[1] = uint8(length >> 16);
	x[2] = uint8(length >> 8);
	x[3] = uint8(length);
	x[4] = m.major;
	x[5] = m.minor;
	copy(x[6:38], m.random);
	x[38] = uint8(len(m.sessionId));
	copy(x[39:39+len(m.sessionId)], m.sessionId);
	y := x[39+len(m.sessionId):];
	y[0] = uint8(len(m.cipherSuites) >> 7);
	y[1] = uint8(len(m.cipherSuites) << 1);
	for i, suite := range m.cipherSuites {
		y[2+i*2] = uint8(suite >> 8);
		y[3+i*2] = uint8(suite);
	}
	z := y[2+len(m.cipherSuites)*2:];
	z[0] = uint8(len(m.compressionMethods));
	copy(z[1:], m.compressionMethods);
	m.raw = x;

	return x;
}

func (m *clientHelloMsg) unmarshal(data []byte) bool {
	if len(data) < 39 {
		return false
	}
	m.raw = data;
	m.major = data[4];
	m.minor = data[5];
	m.random = data[6:38];
	sessionIdLen := int(data[38]);
	if sessionIdLen > 32 || len(data) < 39+sessionIdLen {
		return false
	}
	m.sessionId = data[39 : 39+sessionIdLen];
	data = data[39+sessionIdLen:];
	if len(data) < 2 {
		return false
	}
	// cipherSuiteLen is the number of bytes of cipher suite numbers. Since
	// they are uint16s, the number must be even.
	cipherSuiteLen := int(data[0])<<8 | int(data[1]);
	if cipherSuiteLen%2 == 1 || len(data) < 2+cipherSuiteLen {
		return false
	}
	numCipherSuites := cipherSuiteLen / 2;
	m.cipherSuites = make([]uint16, numCipherSuites);
	for i := 0; i < numCipherSuites; i++ {
		m.cipherSuites[i] = uint16(data[2+2*i])<<8 | uint16(data[3+2*i])
	}
	data = data[2+cipherSuiteLen:];
	if len(data) < 2 {
		return false
	}
	compressionMethodsLen := int(data[0]);
	if len(data) < 1+compressionMethodsLen {
		return false
	}
	m.compressionMethods = data[1 : 1+compressionMethodsLen];

	// A ClientHello may be following by trailing data: RFC 4346 section 7.4.1.2
	return true;
}

type serverHelloMsg struct {
	raw			[]byte;
	major, minor		uint8;
	random			[]byte;
	sessionId		[]byte;
	cipherSuite		uint16;
	compressionMethod	uint8;
}

func (m *serverHelloMsg) marshal() []byte {
	if m.raw != nil {
		return m.raw
	}

	length := 38 + len(m.sessionId);
	x := make([]byte, 4+length);
	x[0] = typeServerHello;
	x[1] = uint8(length >> 16);
	x[2] = uint8(length >> 8);
	x[3] = uint8(length);
	x[4] = m.major;
	x[5] = m.minor;
	copy(x[6:38], m.random);
	x[38] = uint8(len(m.sessionId));
	copy(x[39:39+len(m.sessionId)], m.sessionId);
	z := x[39+len(m.sessionId):];
	z[0] = uint8(m.cipherSuite >> 8);
	z[1] = uint8(m.cipherSuite);
	z[2] = uint8(m.compressionMethod);
	m.raw = x;

	return x;
}

type certificateMsg struct {
	raw		[]byte;
	certificates	[][]byte;
}

func (m *certificateMsg) marshal() (x []byte) {
	if m.raw != nil {
		return m.raw
	}

	var i int;
	for _, slice := range m.certificates {
		i += len(slice)
	}

	length := 3 + 3*len(m.certificates) + i;
	x = make([]byte, 4+length);
	x[0] = typeCertificate;
	x[1] = uint8(length >> 16);
	x[2] = uint8(length >> 8);
	x[3] = uint8(length);

	certificateOctets := length - 3;
	x[4] = uint8(certificateOctets >> 16);
	x[5] = uint8(certificateOctets >> 8);
	x[6] = uint8(certificateOctets);

	y := x[7:];
	for _, slice := range m.certificates {
		y[0] = uint8(len(slice) >> 16);
		y[1] = uint8(len(slice) >> 8);
		y[2] = uint8(len(slice));
		copy(y[3:], slice);
		y = y[3+len(slice):];
	}

	m.raw = x;
	return;
}

type serverHelloDoneMsg struct{}

func (m *serverHelloDoneMsg) marshal() []byte {
	x := make([]byte, 4);
	x[0] = typeServerHelloDone;
	return x;
}

type clientKeyExchangeMsg struct {
	raw		[]byte;
	ciphertext	[]byte;
}

func (m *clientKeyExchangeMsg) marshal() []byte {
	if m.raw != nil {
		return m.raw
	}
	length := len(m.ciphertext) + 2;
	x := make([]byte, length+4);
	x[0] = typeClientKeyExchange;
	x[1] = uint8(length >> 16);
	x[2] = uint8(length >> 8);
	x[3] = uint8(length);
	x[4] = uint8(len(m.ciphertext) >> 8);
	x[5] = uint8(len(m.ciphertext));
	copy(x[6:], m.ciphertext);

	m.raw = x;
	return x;
}

func (m *clientKeyExchangeMsg) unmarshal(data []byte) bool {
	m.raw = data;
	if len(data) < 7 {
		return false
	}
	cipherTextLen := int(data[4])<<8 | int(data[5]);
	if len(data) != 6+cipherTextLen {
		return false
	}
	m.ciphertext = data[6:];
	return true;
}

type finishedMsg struct {
	raw		[]byte;
	verifyData	[]byte;
}

func (m *finishedMsg) marshal() (x []byte) {
	if m.raw != nil {
		return m.raw
	}

	x = make([]byte, 16);
	x[0] = typeFinished;
	x[3] = 12;
	copy(x[4:], m.verifyData);
	m.raw = x;
	return;
}

func (m *finishedMsg) unmarshal(data []byte) bool {
	m.raw = data;
	if len(data) != 4+12 {
		return false
	}
	m.verifyData = data[4:];
	return true;
}
