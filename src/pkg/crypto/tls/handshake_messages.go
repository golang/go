// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

type clientHelloMsg struct {
	raw                []byte
	vers               uint16
	random             []byte
	sessionId          []byte
	cipherSuites       []uint16
	compressionMethods []uint8
	nextProtoNeg       bool
	serverName         string
	ocspStapling       bool
	supportedCurves    []uint16
	supportedPoints    []uint8
}

func (m *clientHelloMsg) marshal() []byte {
	if m.raw != nil {
		return m.raw
	}

	length := 2 + 32 + 1 + len(m.sessionId) + 2 + len(m.cipherSuites)*2 + 1 + len(m.compressionMethods)
	numExtensions := 0
	extensionsLength := 0
	if m.nextProtoNeg {
		numExtensions++
	}
	if m.ocspStapling {
		extensionsLength += 1 + 2 + 2
		numExtensions++
	}
	if len(m.serverName) > 0 {
		extensionsLength += 5 + len(m.serverName)
		numExtensions++
	}
	if len(m.supportedCurves) > 0 {
		extensionsLength += 2 + 2*len(m.supportedCurves)
		numExtensions++
	}
	if len(m.supportedPoints) > 0 {
		extensionsLength += 1 + len(m.supportedPoints)
		numExtensions++
	}
	if numExtensions > 0 {
		extensionsLength += 4 * numExtensions
		length += 2 + extensionsLength
	}

	x := make([]byte, 4+length)
	x[0] = typeClientHello
	x[1] = uint8(length >> 16)
	x[2] = uint8(length >> 8)
	x[3] = uint8(length)
	x[4] = uint8(m.vers >> 8)
	x[5] = uint8(m.vers)
	copy(x[6:38], m.random)
	x[38] = uint8(len(m.sessionId))
	copy(x[39:39+len(m.sessionId)], m.sessionId)
	y := x[39+len(m.sessionId):]
	y[0] = uint8(len(m.cipherSuites) >> 7)
	y[1] = uint8(len(m.cipherSuites) << 1)
	for i, suite := range m.cipherSuites {
		y[2+i*2] = uint8(suite >> 8)
		y[3+i*2] = uint8(suite)
	}
	z := y[2+len(m.cipherSuites)*2:]
	z[0] = uint8(len(m.compressionMethods))
	copy(z[1:], m.compressionMethods)

	z = z[1+len(m.compressionMethods):]
	if numExtensions > 0 {
		z[0] = byte(extensionsLength >> 8)
		z[1] = byte(extensionsLength)
		z = z[2:]
	}
	if m.nextProtoNeg {
		z[0] = byte(extensionNextProtoNeg >> 8)
		z[1] = byte(extensionNextProtoNeg)
		// The length is always 0
		z = z[4:]
	}
	if len(m.serverName) > 0 {
		z[0] = byte(extensionServerName >> 8)
		z[1] = byte(extensionServerName)
		l := len(m.serverName) + 5
		z[2] = byte(l >> 8)
		z[3] = byte(l)
		z = z[4:]

		// RFC 3546, section 3.1
		//
		// struct {
		//     NameType name_type;
		//     select (name_type) {
		//         case host_name: HostName;
		//     } name;
		// } ServerName;
		//
		// enum {
		//     host_name(0), (255)
		// } NameType;
		//
		// opaque HostName<1..2^16-1>;
		//
		// struct {
		//     ServerName server_name_list<1..2^16-1>
		// } ServerNameList;

		z[0] = byte((len(m.serverName) + 3) >> 8)
		z[1] = byte(len(m.serverName) + 3)
		z[3] = byte(len(m.serverName) >> 8)
		z[4] = byte(len(m.serverName))
		copy(z[5:], []byte(m.serverName))
		z = z[l:]
	}
	if m.ocspStapling {
		// RFC 4366, section 3.6
		z[0] = byte(extensionStatusRequest >> 8)
		z[1] = byte(extensionStatusRequest)
		z[2] = 0
		z[3] = 5
		z[4] = 1 // OCSP type
		// Two zero valued uint16s for the two lengths.
		z = z[9:]
	}
	if len(m.supportedCurves) > 0 {
		// http://tools.ietf.org/html/rfc4492#section-5.5.1
		z[0] = byte(extensionSupportedCurves >> 8)
		z[1] = byte(extensionSupportedCurves)
		l := 2 + 2*len(m.supportedCurves)
		z[2] = byte(l >> 8)
		z[3] = byte(l)
		l -= 2
		z[4] = byte(l >> 8)
		z[5] = byte(l)
		z = z[6:]
		for _, curve := range m.supportedCurves {
			z[0] = byte(curve >> 8)
			z[1] = byte(curve)
			z = z[2:]
		}
	}
	if len(m.supportedPoints) > 0 {
		// http://tools.ietf.org/html/rfc4492#section-5.5.2
		z[0] = byte(extensionSupportedPoints >> 8)
		z[1] = byte(extensionSupportedPoints)
		l := 1 + len(m.supportedPoints)
		z[2] = byte(l >> 8)
		z[3] = byte(l)
		l--
		z[4] = byte(l)
		z = z[5:]
		for _, pointFormat := range m.supportedPoints {
			z[0] = byte(pointFormat)
			z = z[1:]
		}
	}

	m.raw = x

	return x
}

func (m *clientHelloMsg) unmarshal(data []byte) bool {
	if len(data) < 42 {
		return false
	}
	m.raw = data
	m.vers = uint16(data[4])<<8 | uint16(data[5])
	m.random = data[6:38]
	sessionIdLen := int(data[38])
	if sessionIdLen > 32 || len(data) < 39+sessionIdLen {
		return false
	}
	m.sessionId = data[39 : 39+sessionIdLen]
	data = data[39+sessionIdLen:]
	if len(data) < 2 {
		return false
	}
	// cipherSuiteLen is the number of bytes of cipher suite numbers. Since
	// they are uint16s, the number must be even.
	cipherSuiteLen := int(data[0])<<8 | int(data[1])
	if cipherSuiteLen%2 == 1 || len(data) < 2+cipherSuiteLen {
		return false
	}
	numCipherSuites := cipherSuiteLen / 2
	m.cipherSuites = make([]uint16, numCipherSuites)
	for i := 0; i < numCipherSuites; i++ {
		m.cipherSuites[i] = uint16(data[2+2*i])<<8 | uint16(data[3+2*i])
	}
	data = data[2+cipherSuiteLen:]
	if len(data) < 1 {
		return false
	}
	compressionMethodsLen := int(data[0])
	if len(data) < 1+compressionMethodsLen {
		return false
	}
	m.compressionMethods = data[1 : 1+compressionMethodsLen]

	data = data[1+compressionMethodsLen:]

	m.nextProtoNeg = false
	m.serverName = ""
	m.ocspStapling = false

	if len(data) == 0 {
		// ClientHello is optionally followed by extension data
		return true
	}
	if len(data) < 2 {
		return false
	}

	extensionsLength := int(data[0])<<8 | int(data[1])
	data = data[2:]
	if extensionsLength != len(data) {
		return false
	}

	for len(data) != 0 {
		if len(data) < 4 {
			return false
		}
		extension := uint16(data[0])<<8 | uint16(data[1])
		length := int(data[2])<<8 | int(data[3])
		data = data[4:]
		if len(data) < length {
			return false
		}

		switch extension {
		case extensionServerName:
			if length < 2 {
				return false
			}
			numNames := int(data[0])<<8 | int(data[1])
			d := data[2:]
			for i := 0; i < numNames; i++ {
				if len(d) < 3 {
					return false
				}
				nameType := d[0]
				nameLen := int(d[1])<<8 | int(d[2])
				d = d[3:]
				if len(d) < nameLen {
					return false
				}
				if nameType == 0 {
					m.serverName = string(d[0:nameLen])
					break
				}
				d = d[nameLen:]
			}
		case extensionNextProtoNeg:
			if length > 0 {
				return false
			}
			m.nextProtoNeg = true
		case extensionStatusRequest:
			m.ocspStapling = length > 0 && data[0] == statusTypeOCSP
		case extensionSupportedCurves:
			// http://tools.ietf.org/html/rfc4492#section-5.5.1
			if length < 2 {
				return false
			}
			l := int(data[0])<<8 | int(data[1])
			if l%2 == 1 || length != l+2 {
				return false
			}
			numCurves := l / 2
			m.supportedCurves = make([]uint16, numCurves)
			d := data[2:]
			for i := 0; i < numCurves; i++ {
				m.supportedCurves[i] = uint16(d[0])<<8 | uint16(d[1])
				d = d[2:]
			}
		case extensionSupportedPoints:
			// http://tools.ietf.org/html/rfc4492#section-5.5.2
			if length < 1 {
				return false
			}
			l := int(data[0])
			if length != l+1 {
				return false
			}
			m.supportedPoints = make([]uint8, l)
			copy(m.supportedPoints, data[1:])
		}
		data = data[length:]
	}

	return true
}

type serverHelloMsg struct {
	raw               []byte
	vers              uint16
	random            []byte
	sessionId         []byte
	cipherSuite       uint16
	compressionMethod uint8
	nextProtoNeg      bool
	nextProtos        []string
	ocspStapling      bool
}

func (m *serverHelloMsg) marshal() []byte {
	if m.raw != nil {
		return m.raw
	}

	length := 38 + len(m.sessionId)
	numExtensions := 0
	extensionsLength := 0

	nextProtoLen := 0
	if m.nextProtoNeg {
		numExtensions++
		for _, v := range m.nextProtos {
			nextProtoLen += len(v)
		}
		nextProtoLen += len(m.nextProtos)
		extensionsLength += nextProtoLen
	}
	if m.ocspStapling {
		numExtensions++
	}
	if numExtensions > 0 {
		extensionsLength += 4 * numExtensions
		length += 2 + extensionsLength
	}

	x := make([]byte, 4+length)
	x[0] = typeServerHello
	x[1] = uint8(length >> 16)
	x[2] = uint8(length >> 8)
	x[3] = uint8(length)
	x[4] = uint8(m.vers >> 8)
	x[5] = uint8(m.vers)
	copy(x[6:38], m.random)
	x[38] = uint8(len(m.sessionId))
	copy(x[39:39+len(m.sessionId)], m.sessionId)
	z := x[39+len(m.sessionId):]
	z[0] = uint8(m.cipherSuite >> 8)
	z[1] = uint8(m.cipherSuite)
	z[2] = uint8(m.compressionMethod)

	z = z[3:]
	if numExtensions > 0 {
		z[0] = byte(extensionsLength >> 8)
		z[1] = byte(extensionsLength)
		z = z[2:]
	}
	if m.nextProtoNeg {
		z[0] = byte(extensionNextProtoNeg >> 8)
		z[1] = byte(extensionNextProtoNeg)
		z[2] = byte(nextProtoLen >> 8)
		z[3] = byte(nextProtoLen)
		z = z[4:]

		for _, v := range m.nextProtos {
			l := len(v)
			if l > 255 {
				l = 255
			}
			z[0] = byte(l)
			copy(z[1:], []byte(v[0:l]))
			z = z[1+l:]
		}
	}
	if m.ocspStapling {
		z[0] = byte(extensionStatusRequest >> 8)
		z[1] = byte(extensionStatusRequest)
		z = z[4:]
	}

	m.raw = x

	return x
}

func (m *serverHelloMsg) unmarshal(data []byte) bool {
	if len(data) < 42 {
		return false
	}
	m.raw = data
	m.vers = uint16(data[4])<<8 | uint16(data[5])
	m.random = data[6:38]
	sessionIdLen := int(data[38])
	if sessionIdLen > 32 || len(data) < 39+sessionIdLen {
		return false
	}
	m.sessionId = data[39 : 39+sessionIdLen]
	data = data[39+sessionIdLen:]
	if len(data) < 3 {
		return false
	}
	m.cipherSuite = uint16(data[0])<<8 | uint16(data[1])
	m.compressionMethod = data[2]
	data = data[3:]

	m.nextProtoNeg = false
	m.nextProtos = nil
	m.ocspStapling = false

	if len(data) == 0 {
		// ServerHello is optionally followed by extension data
		return true
	}
	if len(data) < 2 {
		return false
	}

	extensionsLength := int(data[0])<<8 | int(data[1])
	data = data[2:]
	if len(data) != extensionsLength {
		return false
	}

	for len(data) != 0 {
		if len(data) < 4 {
			return false
		}
		extension := uint16(data[0])<<8 | uint16(data[1])
		length := int(data[2])<<8 | int(data[3])
		data = data[4:]
		if len(data) < length {
			return false
		}

		switch extension {
		case extensionNextProtoNeg:
			m.nextProtoNeg = true
			d := data
			for len(d) > 0 {
				l := int(d[0])
				d = d[1:]
				if l == 0 || l > len(d) {
					return false
				}
				m.nextProtos = append(m.nextProtos, string(d[0:l]))
				d = d[l:]
			}
		case extensionStatusRequest:
			if length > 0 {
				return false
			}
			m.ocspStapling = true
		}
		data = data[length:]
	}

	return true
}

type certificateMsg struct {
	raw          []byte
	certificates [][]byte
}

func (m *certificateMsg) marshal() (x []byte) {
	if m.raw != nil {
		return m.raw
	}

	var i int
	for _, slice := range m.certificates {
		i += len(slice)
	}

	length := 3 + 3*len(m.certificates) + i
	x = make([]byte, 4+length)
	x[0] = typeCertificate
	x[1] = uint8(length >> 16)
	x[2] = uint8(length >> 8)
	x[3] = uint8(length)

	certificateOctets := length - 3
	x[4] = uint8(certificateOctets >> 16)
	x[5] = uint8(certificateOctets >> 8)
	x[6] = uint8(certificateOctets)

	y := x[7:]
	for _, slice := range m.certificates {
		y[0] = uint8(len(slice) >> 16)
		y[1] = uint8(len(slice) >> 8)
		y[2] = uint8(len(slice))
		copy(y[3:], slice)
		y = y[3+len(slice):]
	}

	m.raw = x
	return
}

func (m *certificateMsg) unmarshal(data []byte) bool {
	if len(data) < 7 {
		return false
	}

	m.raw = data
	certsLen := uint32(data[4])<<16 | uint32(data[5])<<8 | uint32(data[6])
	if uint32(len(data)) != certsLen+7 {
		return false
	}

	numCerts := 0
	d := data[7:]
	for certsLen > 0 {
		if len(d) < 4 {
			return false
		}
		certLen := uint32(d[0])<<24 | uint32(d[1])<<8 | uint32(d[2])
		if uint32(len(d)) < 3+certLen {
			return false
		}
		d = d[3+certLen:]
		certsLen -= 3 + certLen
		numCerts++
	}

	m.certificates = make([][]byte, numCerts)
	d = data[7:]
	for i := 0; i < numCerts; i++ {
		certLen := uint32(d[0])<<24 | uint32(d[1])<<8 | uint32(d[2])
		m.certificates[i] = d[3 : 3+certLen]
		d = d[3+certLen:]
	}

	return true
}

type serverKeyExchangeMsg struct {
	raw []byte
	key []byte
}

func (m *serverKeyExchangeMsg) marshal() []byte {
	if m.raw != nil {
		return m.raw
	}
	length := len(m.key)
	x := make([]byte, length+4)
	x[0] = typeServerKeyExchange
	x[1] = uint8(length >> 16)
	x[2] = uint8(length >> 8)
	x[3] = uint8(length)
	copy(x[4:], m.key)

	m.raw = x
	return x
}

func (m *serverKeyExchangeMsg) unmarshal(data []byte) bool {
	m.raw = data
	if len(data) < 4 {
		return false
	}
	m.key = data[4:]
	return true
}

type certificateStatusMsg struct {
	raw        []byte
	statusType uint8
	response   []byte
}

func (m *certificateStatusMsg) marshal() []byte {
	if m.raw != nil {
		return m.raw
	}

	var x []byte
	if m.statusType == statusTypeOCSP {
		x = make([]byte, 4+4+len(m.response))
		x[0] = typeCertificateStatus
		l := len(m.response) + 4
		x[1] = byte(l >> 16)
		x[2] = byte(l >> 8)
		x[3] = byte(l)
		x[4] = statusTypeOCSP

		l -= 4
		x[5] = byte(l >> 16)
		x[6] = byte(l >> 8)
		x[7] = byte(l)
		copy(x[8:], m.response)
	} else {
		x = []byte{typeCertificateStatus, 0, 0, 1, m.statusType}
	}

	m.raw = x
	return x
}

func (m *certificateStatusMsg) unmarshal(data []byte) bool {
	m.raw = data
	if len(data) < 5 {
		return false
	}
	m.statusType = data[4]

	m.response = nil
	if m.statusType == statusTypeOCSP {
		if len(data) < 8 {
			return false
		}
		respLen := uint32(data[5])<<16 | uint32(data[6])<<8 | uint32(data[7])
		if uint32(len(data)) != 4+4+respLen {
			return false
		}
		m.response = data[8:]
	}
	return true
}

type serverHelloDoneMsg struct{}

func (m *serverHelloDoneMsg) marshal() []byte {
	x := make([]byte, 4)
	x[0] = typeServerHelloDone
	return x
}

func (m *serverHelloDoneMsg) unmarshal(data []byte) bool {
	return len(data) == 4
}

type clientKeyExchangeMsg struct {
	raw        []byte
	ciphertext []byte
}

func (m *clientKeyExchangeMsg) marshal() []byte {
	if m.raw != nil {
		return m.raw
	}
	length := len(m.ciphertext)
	x := make([]byte, length+4)
	x[0] = typeClientKeyExchange
	x[1] = uint8(length >> 16)
	x[2] = uint8(length >> 8)
	x[3] = uint8(length)
	copy(x[4:], m.ciphertext)

	m.raw = x
	return x
}

func (m *clientKeyExchangeMsg) unmarshal(data []byte) bool {
	m.raw = data
	if len(data) < 4 {
		return false
	}
	l := int(data[1])<<16 | int(data[2])<<8 | int(data[3])
	if l != len(data)-4 {
		return false
	}
	m.ciphertext = data[4:]
	return true
}

type finishedMsg struct {
	raw        []byte
	verifyData []byte
}

func (m *finishedMsg) marshal() (x []byte) {
	if m.raw != nil {
		return m.raw
	}

	x = make([]byte, 4+len(m.verifyData))
	x[0] = typeFinished
	x[3] = byte(len(m.verifyData))
	copy(x[4:], m.verifyData)
	m.raw = x
	return
}

func (m *finishedMsg) unmarshal(data []byte) bool {
	m.raw = data
	if len(data) < 4 {
		return false
	}
	m.verifyData = data[4:]
	return true
}

type nextProtoMsg struct {
	raw   []byte
	proto string
}

func (m *nextProtoMsg) marshal() []byte {
	if m.raw != nil {
		return m.raw
	}
	l := len(m.proto)
	if l > 255 {
		l = 255
	}

	padding := 32 - (l+2)%32
	length := l + padding + 2
	x := make([]byte, length+4)
	x[0] = typeNextProtocol
	x[1] = uint8(length >> 16)
	x[2] = uint8(length >> 8)
	x[3] = uint8(length)

	y := x[4:]
	y[0] = byte(l)
	copy(y[1:], []byte(m.proto[0:l]))
	y = y[1+l:]
	y[0] = byte(padding)

	m.raw = x

	return x
}

func (m *nextProtoMsg) unmarshal(data []byte) bool {
	m.raw = data

	if len(data) < 5 {
		return false
	}
	data = data[4:]
	protoLen := int(data[0])
	data = data[1:]
	if len(data) < protoLen {
		return false
	}
	m.proto = string(data[0:protoLen])
	data = data[protoLen:]

	if len(data) < 1 {
		return false
	}
	paddingLen := int(data[0])
	data = data[1:]
	if len(data) != paddingLen {
		return false
	}

	return true
}

type certificateRequestMsg struct {
	raw                    []byte
	certificateTypes       []byte
	certificateAuthorities [][]byte
}

func (m *certificateRequestMsg) marshal() (x []byte) {
	if m.raw != nil {
		return m.raw
	}

	// See http://tools.ietf.org/html/rfc4346#section-7.4.4
	length := 1 + len(m.certificateTypes) + 2
	for _, ca := range m.certificateAuthorities {
		length += 2 + len(ca)
	}

	x = make([]byte, 4+length)
	x[0] = typeCertificateRequest
	x[1] = uint8(length >> 16)
	x[2] = uint8(length >> 8)
	x[3] = uint8(length)

	x[4] = uint8(len(m.certificateTypes))

	copy(x[5:], m.certificateTypes)
	y := x[5+len(m.certificateTypes):]

	numCA := len(m.certificateAuthorities)
	y[0] = uint8(numCA >> 8)
	y[1] = uint8(numCA)
	y = y[2:]
	for _, ca := range m.certificateAuthorities {
		y[0] = uint8(len(ca) >> 8)
		y[1] = uint8(len(ca))
		y = y[2:]
		copy(y, ca)
		y = y[len(ca):]
	}

	m.raw = x

	return
}

func (m *certificateRequestMsg) unmarshal(data []byte) bool {
	m.raw = data

	if len(data) < 5 {
		return false
	}

	length := uint32(data[1])<<16 | uint32(data[2])<<8 | uint32(data[3])
	if uint32(len(data))-4 != length {
		return false
	}

	numCertTypes := int(data[4])
	data = data[5:]
	if numCertTypes == 0 || len(data) <= numCertTypes {
		return false
	}

	m.certificateTypes = make([]byte, numCertTypes)
	if copy(m.certificateTypes, data) != numCertTypes {
		return false
	}

	data = data[numCertTypes:]
	if len(data) < 2 {
		return false
	}

	numCAs := uint16(data[0])<<16 | uint16(data[1])
	data = data[2:]

	m.certificateAuthorities = make([][]byte, numCAs)
	for i := uint16(0); i < numCAs; i++ {
		if len(data) < 2 {
			return false
		}
		caLen := uint16(data[0])<<16 | uint16(data[1])

		data = data[2:]
		if len(data) < int(caLen) {
			return false
		}

		ca := make([]byte, caLen)
		copy(ca, data)
		m.certificateAuthorities[i] = ca
		data = data[caLen:]
	}

	if len(data) > 0 {
		return false
	}

	return true
}

type certificateVerifyMsg struct {
	raw       []byte
	signature []byte
}

func (m *certificateVerifyMsg) marshal() (x []byte) {
	if m.raw != nil {
		return m.raw
	}

	// See http://tools.ietf.org/html/rfc4346#section-7.4.8
	siglength := len(m.signature)
	length := 2 + siglength
	x = make([]byte, 4+length)
	x[0] = typeCertificateVerify
	x[1] = uint8(length >> 16)
	x[2] = uint8(length >> 8)
	x[3] = uint8(length)
	x[4] = uint8(siglength >> 8)
	x[5] = uint8(siglength)
	copy(x[6:], m.signature)

	m.raw = x

	return
}

func (m *certificateVerifyMsg) unmarshal(data []byte) bool {
	m.raw = data

	if len(data) < 6 {
		return false
	}

	length := uint32(data[1])<<16 | uint32(data[2])<<8 | uint32(data[3])
	if uint32(len(data))-4 != length {
		return false
	}

	siglength := int(data[4])<<8 + int(data[5])
	if len(data)-6 != siglength {
		return false
	}

	m.signature = data[6:]

	return true
}
