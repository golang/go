// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"fmt"
	"golang_org/x/crypto/cryptobyte"
	"strings"
)

// The marshalingFunction type is an adapter to allow the use of ordinary
// functions as cryptobyte.MarshalingValue.
type marshalingFunction func(b *cryptobyte.Builder) error

func (f marshalingFunction) Marshal(b *cryptobyte.Builder) error {
	return f(b)
}

// addBytesWithLength appends a sequence of bytes to the cryptobyte.Builder. If
// the length of the sequence is not the value specified, it produces an error.
func addBytesWithLength(b *cryptobyte.Builder, v []byte, n int) {
	b.AddValue(marshalingFunction(func(b *cryptobyte.Builder) error {
		if len(v) != n {
			return fmt.Errorf("invalid value length: expected %d, got %d", n, len(v))
		}
		b.AddBytes(v)
		return nil
	}))
}

// readUint8LengthPrefixed acts like s.ReadUint8LengthPrefixed, but targets a
// []byte instead of a cryptobyte.String.
func readUint8LengthPrefixed(s *cryptobyte.String, out *[]byte) bool {
	return s.ReadUint8LengthPrefixed((*cryptobyte.String)(out))
}

// readUint16LengthPrefixed acts like s.ReadUint16LengthPrefixed, but targets a
// []byte instead of a cryptobyte.String.
func readUint16LengthPrefixed(s *cryptobyte.String, out *[]byte) bool {
	return s.ReadUint16LengthPrefixed((*cryptobyte.String)(out))
}

// readUint24LengthPrefixed acts like s.ReadUint24LengthPrefixed, but targets a
// []byte instead of a cryptobyte.String.
func readUint24LengthPrefixed(s *cryptobyte.String, out *[]byte) bool {
	return s.ReadUint24LengthPrefixed((*cryptobyte.String)(out))
}

type clientHelloMsg struct {
	raw                              []byte
	vers                             uint16
	random                           []byte
	sessionId                        []byte
	cipherSuites                     []uint16
	compressionMethods               []uint8
	nextProtoNeg                     bool
	serverName                       string
	ocspStapling                     bool
	supportedCurves                  []CurveID
	supportedPoints                  []uint8
	ticketSupported                  bool
	sessionTicket                    []uint8
	supportedSignatureAlgorithms     []SignatureScheme
	supportedSignatureAlgorithmsCert []SignatureScheme
	secureRenegotiationSupported     bool
	secureRenegotiation              []byte
	alpnProtocols                    []string
	scts                             bool
	supportedVersions                []uint16
	cookie                           []byte
	keyShares                        []keyShare
	pskModes                         []uint8
	pskIdentities                    []pskIdentity
	pskBinders                       [][]byte
}

func (m *clientHelloMsg) marshal() []byte {
	if m.raw != nil {
		return m.raw
	}

	var b cryptobyte.Builder
	b.AddUint8(typeClientHello)
	b.AddUint24LengthPrefixed(func(b *cryptobyte.Builder) {
		b.AddUint16(m.vers)
		addBytesWithLength(b, m.random, 32)
		b.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) {
			b.AddBytes(m.sessionId)
		})
		b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
			for _, suite := range m.cipherSuites {
				b.AddUint16(suite)
			}
		})
		b.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) {
			b.AddBytes(m.compressionMethods)
		})

		// If extensions aren't present, omit them.
		var extensionsPresent bool
		bWithoutExtensions := *b

		b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
			if m.nextProtoNeg {
				// draft-agl-tls-nextprotoneg-04
				b.AddUint16(extensionNextProtoNeg)
				b.AddUint16(0) // empty extension_data
			}
			if len(m.serverName) > 0 {
				// RFC 6066, Section 3
				b.AddUint16(extensionServerName)
				b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
					b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
						b.AddUint8(0) // name_type = host_name
						b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
							b.AddBytes([]byte(m.serverName))
						})
					})
				})
			}
			if m.ocspStapling {
				// RFC 4366, Section 3.6
				b.AddUint16(extensionStatusRequest)
				b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
					b.AddUint8(1)  // status_type = ocsp
					b.AddUint16(0) // empty responder_id_list
					b.AddUint16(0) // empty request_extensions
				})
			}
			if len(m.supportedCurves) > 0 {
				// RFC 4492, Section 5.1.1 and RFC 8446, Section 4.2.7
				b.AddUint16(extensionSupportedCurves)
				b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
					b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
						for _, curve := range m.supportedCurves {
							b.AddUint16(uint16(curve))
						}
					})
				})
			}
			if len(m.supportedPoints) > 0 {
				// RFC 4492, Section 5.1.2
				b.AddUint16(extensionSupportedPoints)
				b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
					b.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) {
						b.AddBytes(m.supportedPoints)
					})
				})
			}
			if m.ticketSupported {
				// RFC 5077, Section 3.2
				b.AddUint16(extensionSessionTicket)
				b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
					b.AddBytes(m.sessionTicket)
				})
			}
			if len(m.supportedSignatureAlgorithms) > 0 {
				// RFC 5246, Section 7.4.1.4.1
				b.AddUint16(extensionSignatureAlgorithms)
				b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
					b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
						for _, sigAlgo := range m.supportedSignatureAlgorithms {
							b.AddUint16(uint16(sigAlgo))
						}
					})
				})
			}
			if len(m.supportedSignatureAlgorithmsCert) > 0 {
				// RFC 8446, Section 4.2.3
				b.AddUint16(extensionSignatureAlgorithmsCert)
				b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
					b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
						for _, sigAlgo := range m.supportedSignatureAlgorithmsCert {
							b.AddUint16(uint16(sigAlgo))
						}
					})
				})
			}
			if m.secureRenegotiationSupported {
				// RFC 5746, Section 3.2
				b.AddUint16(extensionRenegotiationInfo)
				b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
					b.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) {
						b.AddBytes(m.secureRenegotiation)
					})
				})
			}
			if len(m.alpnProtocols) > 0 {
				// RFC 7301, Section 3.1
				b.AddUint16(extensionALPN)
				b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
					b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
						for _, proto := range m.alpnProtocols {
							b.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) {
								b.AddBytes([]byte(proto))
							})
						}
					})
				})
			}
			if m.scts {
				// RFC 6962, Section 3.3.1
				b.AddUint16(extensionSCT)
				b.AddUint16(0) // empty extension_data
			}
			if len(m.supportedVersions) > 0 {
				// RFC 8446, Section 4.2.1
				b.AddUint16(extensionSupportedVersions)
				b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
					b.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) {
						for _, vers := range m.supportedVersions {
							b.AddUint16(vers)
						}
					})
				})
			}
			if len(m.cookie) > 0 {
				// RFC 8446, Section 4.2.2
				b.AddUint16(extensionCookie)
				b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
					b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
						b.AddBytes(m.cookie)
					})
				})
			}
			if len(m.keyShares) > 0 {
				// RFC 8446, Section 4.2.8
				b.AddUint16(extensionKeyShare)
				b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
					b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
						for _, ks := range m.keyShares {
							b.AddUint16(uint16(ks.group))
							b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
								b.AddBytes(ks.data)
							})
						}
					})
				})
			}
			if len(m.pskModes) > 0 {
				// RFC 8446, Section 4.2.9
				b.AddUint16(extensionPSKModes)
				b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
					b.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) {
						b.AddBytes(m.pskModes)
					})
				})
			}
			if len(m.pskIdentities) > 0 { // pre_shared_key must be the last extension
				// RFC 8446, Section 4.2.11
				b.AddUint16(extensionPreSharedKey)
				b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
					b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
						for _, psk := range m.pskIdentities {
							b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
								b.AddBytes(psk.label)
							})
							b.AddUint32(psk.obfuscatedTicketAge)
						}
					})
					b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
						for _, binder := range m.pskBinders {
							b.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) {
								b.AddBytes(binder)
							})
						}
					})
				})
			}

			extensionsPresent = len(b.BytesOrPanic()) > 2
		})

		if !extensionsPresent {
			*b = bWithoutExtensions
		}
	})

	m.raw = b.BytesOrPanic()
	return m.raw
}

func (m *clientHelloMsg) unmarshal(data []byte) bool {
	*m = clientHelloMsg{raw: data}
	s := cryptobyte.String(data)

	if !s.Skip(4) || // message type and uint24 length field
		!s.ReadUint16(&m.vers) || !s.ReadBytes(&m.random, 32) ||
		!readUint8LengthPrefixed(&s, &m.sessionId) {
		return false
	}

	var cipherSuites cryptobyte.String
	if !s.ReadUint16LengthPrefixed(&cipherSuites) {
		return false
	}
	m.cipherSuites = []uint16{}
	m.secureRenegotiationSupported = false
	for !cipherSuites.Empty() {
		var suite uint16
		if !cipherSuites.ReadUint16(&suite) {
			return false
		}
		if suite == scsvRenegotiation {
			m.secureRenegotiationSupported = true
		}
		m.cipherSuites = append(m.cipherSuites, suite)
	}

	if !readUint8LengthPrefixed(&s, &m.compressionMethods) {
		return false
	}

	if s.Empty() {
		// ClientHello is optionally followed by extension data
		return true
	}

	var extensions cryptobyte.String
	if !s.ReadUint16LengthPrefixed(&extensions) || !s.Empty() {
		return false
	}

	for !extensions.Empty() {
		var extension uint16
		var extData cryptobyte.String
		if !extensions.ReadUint16(&extension) ||
			!extensions.ReadUint16LengthPrefixed(&extData) {
			return false
		}

		switch extension {
		case extensionServerName:
			// RFC 6066, Section 3
			var nameList cryptobyte.String
			if !extData.ReadUint16LengthPrefixed(&nameList) || nameList.Empty() {
				return false
			}
			for !nameList.Empty() {
				var nameType uint8
				var serverName cryptobyte.String
				if !nameList.ReadUint8(&nameType) ||
					!nameList.ReadUint16LengthPrefixed(&serverName) ||
					serverName.Empty() {
					return false
				}
				if nameType != 0 {
					continue
				}
				if len(m.serverName) != 0 {
					// Multiple names of the same name_type are prohibited.
					return false
				}
				m.serverName = string(serverName)
				// An SNI value may not include a trailing dot.
				if strings.HasSuffix(m.serverName, ".") {
					return false
				}
			}
		case extensionNextProtoNeg:
			// draft-agl-tls-nextprotoneg-04
			m.nextProtoNeg = true
		case extensionStatusRequest:
			// RFC 4366, Section 3.6
			var statusType uint8
			var ignored cryptobyte.String
			if !extData.ReadUint8(&statusType) ||
				!extData.ReadUint16LengthPrefixed(&ignored) ||
				!extData.ReadUint16LengthPrefixed(&ignored) {
				return false
			}
			m.ocspStapling = statusType == statusTypeOCSP
		case extensionSupportedCurves:
			// RFC 4492, Section 5.1.1 and RFC 8446, Section 4.2.7
			var curves cryptobyte.String
			if !extData.ReadUint16LengthPrefixed(&curves) || curves.Empty() {
				return false
			}
			for !curves.Empty() {
				var curve uint16
				if !curves.ReadUint16(&curve) {
					return false
				}
				m.supportedCurves = append(m.supportedCurves, CurveID(curve))
			}
		case extensionSupportedPoints:
			// RFC 4492, Section 5.1.2
			if !readUint8LengthPrefixed(&extData, &m.supportedPoints) ||
				len(m.supportedPoints) == 0 {
				return false
			}
		case extensionSessionTicket:
			// RFC 5077, Section 3.2
			m.ticketSupported = true
			extData.ReadBytes(&m.sessionTicket, len(extData))
		case extensionSignatureAlgorithms:
			// RFC 5246, Section 7.4.1.4.1
			var sigAndAlgs cryptobyte.String
			if !extData.ReadUint16LengthPrefixed(&sigAndAlgs) || sigAndAlgs.Empty() {
				return false
			}
			for !sigAndAlgs.Empty() {
				var sigAndAlg uint16
				if !sigAndAlgs.ReadUint16(&sigAndAlg) {
					return false
				}
				m.supportedSignatureAlgorithms = append(
					m.supportedSignatureAlgorithms, SignatureScheme(sigAndAlg))
			}
		case extensionSignatureAlgorithmsCert:
			// RFC 8446, Section 4.2.3
			var sigAndAlgs cryptobyte.String
			if !extData.ReadUint16LengthPrefixed(&sigAndAlgs) || sigAndAlgs.Empty() {
				return false
			}
			for !sigAndAlgs.Empty() {
				var sigAndAlg uint16
				if !sigAndAlgs.ReadUint16(&sigAndAlg) {
					return false
				}
				m.supportedSignatureAlgorithmsCert = append(
					m.supportedSignatureAlgorithmsCert, SignatureScheme(sigAndAlg))
			}
		case extensionRenegotiationInfo:
			// RFC 5746, Section 3.2
			if !readUint8LengthPrefixed(&extData, &m.secureRenegotiation) {
				return false
			}
			m.secureRenegotiationSupported = true
		case extensionALPN:
			// RFC 7301, Section 3.1
			var protoList cryptobyte.String
			if !extData.ReadUint16LengthPrefixed(&protoList) || protoList.Empty() {
				return false
			}
			for !protoList.Empty() {
				var proto cryptobyte.String
				if !protoList.ReadUint8LengthPrefixed(&proto) || proto.Empty() {
					return false
				}
				m.alpnProtocols = append(m.alpnProtocols, string(proto))
			}
		case extensionSCT:
			// RFC 6962, Section 3.3.1
			m.scts = true
		case extensionSupportedVersions:
			// RFC 8446, Section 4.2.1
			var versList cryptobyte.String
			if !extData.ReadUint8LengthPrefixed(&versList) || versList.Empty() {
				return false
			}
			for !versList.Empty() {
				var vers uint16
				if !versList.ReadUint16(&vers) {
					return false
				}
				m.supportedVersions = append(m.supportedVersions, vers)
			}
		case extensionCookie:
			// RFC 8446, Section 4.2.2
			if !readUint16LengthPrefixed(&extData, &m.cookie) {
				return false
			}
		case extensionKeyShare:
			// RFC 8446, Section 4.2.8
			var clientShares cryptobyte.String
			if !extData.ReadUint16LengthPrefixed(&clientShares) {
				return false
			}
			for !clientShares.Empty() {
				var ks keyShare
				if !clientShares.ReadUint16((*uint16)(&ks.group)) ||
					!readUint16LengthPrefixed(&clientShares, &ks.data) ||
					len(ks.data) == 0 {
					return false
				}
				m.keyShares = append(m.keyShares, ks)
			}
		case extensionPSKModes:
			// RFC 8446, Section 4.2.9
			if !readUint8LengthPrefixed(&extData, &m.pskModes) {
				return false
			}
		case extensionPreSharedKey:
			// RFC 8446, Section 4.2.11
			if !extensions.Empty() {
				return false // pre_shared_key must be the last extension
			}
			var identities cryptobyte.String
			if !extData.ReadUint16LengthPrefixed(&identities) || identities.Empty() {
				return false
			}
			for !identities.Empty() {
				var psk pskIdentity
				if !readUint16LengthPrefixed(&identities, &psk.label) ||
					!identities.ReadUint32(&psk.obfuscatedTicketAge) ||
					len(psk.label) == 0 {
					return false
				}
				m.pskIdentities = append(m.pskIdentities, psk)
			}
			var binders cryptobyte.String
			if !extData.ReadUint16LengthPrefixed(&binders) || binders.Empty() {
				return false
			}
			for !binders.Empty() {
				var binder []byte
				if !readUint8LengthPrefixed(&binders, &binder) ||
					len(binder) == 0 {
					return false
				}
				m.pskBinders = append(m.pskBinders, binder)
			}
		default:
			// Ignore unknown extensions.
			continue
		}

		if !extData.Empty() {
			return false
		}
	}

	return true
}

type serverHelloMsg struct {
	raw                          []byte
	vers                         uint16
	random                       []byte
	sessionId                    []byte
	cipherSuite                  uint16
	compressionMethod            uint8
	nextProtoNeg                 bool
	nextProtos                   []string
	ocspStapling                 bool
	ticketSupported              bool
	secureRenegotiationSupported bool
	secureRenegotiation          []byte
	alpnProtocol                 string
	scts                         [][]byte
	supportedVersion             uint16
	serverShare                  keyShare
	selectedIdentityPresent      bool
	selectedIdentity             uint16

	// HelloRetryRequest extensions
	cookie        []byte
	selectedGroup CurveID
}

func (m *serverHelloMsg) marshal() []byte {
	if m.raw != nil {
		return m.raw
	}

	var b cryptobyte.Builder
	b.AddUint8(typeServerHello)
	b.AddUint24LengthPrefixed(func(b *cryptobyte.Builder) {
		b.AddUint16(m.vers)
		addBytesWithLength(b, m.random, 32)
		b.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) {
			b.AddBytes(m.sessionId)
		})
		b.AddUint16(m.cipherSuite)
		b.AddUint8(m.compressionMethod)

		// If extensions aren't present, omit them.
		var extensionsPresent bool
		bWithoutExtensions := *b

		b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
			if m.nextProtoNeg {
				b.AddUint16(extensionNextProtoNeg)
				b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
					for _, proto := range m.nextProtos {
						b.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) {
							b.AddBytes([]byte(proto))
						})
					}
				})
			}
			if m.ocspStapling {
				b.AddUint16(extensionStatusRequest)
				b.AddUint16(0) // empty extension_data
			}
			if m.ticketSupported {
				b.AddUint16(extensionSessionTicket)
				b.AddUint16(0) // empty extension_data
			}
			if m.secureRenegotiationSupported {
				b.AddUint16(extensionRenegotiationInfo)
				b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
					b.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) {
						b.AddBytes(m.secureRenegotiation)
					})
				})
			}
			if len(m.alpnProtocol) > 0 {
				b.AddUint16(extensionALPN)
				b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
					b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
						b.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) {
							b.AddBytes([]byte(m.alpnProtocol))
						})
					})
				})
			}
			if len(m.scts) > 0 {
				b.AddUint16(extensionSCT)
				b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
					b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
						for _, sct := range m.scts {
							b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
								b.AddBytes(sct)
							})
						}
					})
				})
			}
			if m.supportedVersion != 0 {
				b.AddUint16(extensionSupportedVersions)
				b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
					b.AddUint16(m.supportedVersion)
				})
			}
			if m.serverShare.group != 0 {
				b.AddUint16(extensionKeyShare)
				b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
					b.AddUint16(uint16(m.serverShare.group))
					b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
						b.AddBytes(m.serverShare.data)
					})
				})
			}
			if m.selectedIdentityPresent {
				b.AddUint16(extensionPreSharedKey)
				b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
					b.AddUint16(m.selectedIdentity)
				})
			}

			if len(m.cookie) > 0 {
				b.AddUint16(extensionCookie)
				b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
					b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
						b.AddBytes(m.cookie)
					})
				})
			}
			if m.selectedGroup != 0 {
				b.AddUint16(extensionKeyShare)
				b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
					b.AddUint16(uint16(m.selectedGroup))
				})
			}

			extensionsPresent = len(b.BytesOrPanic()) > 2
		})

		if !extensionsPresent {
			*b = bWithoutExtensions
		}
	})

	m.raw = b.BytesOrPanic()
	return m.raw
}

func (m *serverHelloMsg) unmarshal(data []byte) bool {
	*m = serverHelloMsg{raw: data}
	s := cryptobyte.String(data)

	if !s.Skip(4) || // message type and uint24 length field
		!s.ReadUint16(&m.vers) || !s.ReadBytes(&m.random, 32) ||
		!readUint8LengthPrefixed(&s, &m.sessionId) ||
		!s.ReadUint16(&m.cipherSuite) ||
		!s.ReadUint8(&m.compressionMethod) {
		return false
	}

	if s.Empty() {
		// ServerHello is optionally followed by extension data
		return true
	}

	var extensions cryptobyte.String
	if !s.ReadUint16LengthPrefixed(&extensions) || !s.Empty() {
		return false
	}

	for !extensions.Empty() {
		var extension uint16
		var extData cryptobyte.String
		if !extensions.ReadUint16(&extension) ||
			!extensions.ReadUint16LengthPrefixed(&extData) {
			return false
		}

		switch extension {
		case extensionNextProtoNeg:
			m.nextProtoNeg = true
			for !extData.Empty() {
				var proto cryptobyte.String
				if !extData.ReadUint8LengthPrefixed(&proto) ||
					proto.Empty() {
					return false
				}
				m.nextProtos = append(m.nextProtos, string(proto))
			}
		case extensionStatusRequest:
			m.ocspStapling = true
		case extensionSessionTicket:
			m.ticketSupported = true
		case extensionRenegotiationInfo:
			if !readUint8LengthPrefixed(&extData, &m.secureRenegotiation) {
				return false
			}
			m.secureRenegotiationSupported = true
		case extensionALPN:
			var protoList cryptobyte.String
			if !extData.ReadUint16LengthPrefixed(&protoList) || protoList.Empty() {
				return false
			}
			var proto cryptobyte.String
			if !protoList.ReadUint8LengthPrefixed(&proto) ||
				proto.Empty() || !protoList.Empty() {
				return false
			}
			m.alpnProtocol = string(proto)
		case extensionSCT:
			var sctList cryptobyte.String
			if !extData.ReadUint16LengthPrefixed(&sctList) || sctList.Empty() {
				return false
			}
			for !sctList.Empty() {
				var sct []byte
				if !readUint16LengthPrefixed(&sctList, &sct) ||
					len(sct) == 0 {
					return false
				}
				m.scts = append(m.scts, sct)
			}
		case extensionSupportedVersions:
			if !extData.ReadUint16(&m.supportedVersion) {
				return false
			}
		case extensionCookie:
			if !readUint16LengthPrefixed(&extData, &m.cookie) {
				return false
			}
		case extensionKeyShare:
			// This extension has different formats in SH and HRR, accept either
			// and let the handshake logic decide. See RFC 8446, Section 4.2.8.
			if len(extData) == 2 {
				if !extData.ReadUint16((*uint16)(&m.selectedGroup)) {
					return false
				}
			} else {
				if !extData.ReadUint16((*uint16)(&m.serverShare.group)) ||
					!readUint16LengthPrefixed(&extData, &m.serverShare.data) {
					return false
				}
			}
		case extensionPreSharedKey:
			m.selectedIdentityPresent = true
			if !extData.ReadUint16(&m.selectedIdentity) {
				return false
			}
		default:
			// Ignore unknown extensions.
			continue
		}

		if !extData.Empty() {
			return false
		}
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
		certLen := uint32(d[0])<<16 | uint32(d[1])<<8 | uint32(d[2])
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
		certLen := uint32(d[0])<<16 | uint32(d[1])<<8 | uint32(d[2])
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

func (m *finishedMsg) marshal() []byte {
	if m.raw != nil {
		return m.raw
	}

	var b cryptobyte.Builder
	b.AddUint8(typeFinished)
	b.AddUint24LengthPrefixed(func(b *cryptobyte.Builder) {
		b.AddBytes(m.verifyData)
	})

	m.raw = b.BytesOrPanic()
	return m.raw
}

func (m *finishedMsg) unmarshal(data []byte) bool {
	m.raw = data
	s := cryptobyte.String(data)
	return s.Skip(1) &&
		readUint24LengthPrefixed(&s, &m.verifyData) &&
		s.Empty()
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
	raw []byte
	// hasSignatureAlgorithm indicates whether this message includes a list of
	// supported signature algorithms. This change was introduced with TLS 1.2.
	hasSignatureAlgorithm bool

	certificateTypes             []byte
	supportedSignatureAlgorithms []SignatureScheme
	certificateAuthorities       [][]byte
}

func (m *certificateRequestMsg) marshal() (x []byte) {
	if m.raw != nil {
		return m.raw
	}

	// See RFC 4346, Section 7.4.4.
	length := 1 + len(m.certificateTypes) + 2
	casLength := 0
	for _, ca := range m.certificateAuthorities {
		casLength += 2 + len(ca)
	}
	length += casLength

	if m.hasSignatureAlgorithm {
		length += 2 + 2*len(m.supportedSignatureAlgorithms)
	}

	x = make([]byte, 4+length)
	x[0] = typeCertificateRequest
	x[1] = uint8(length >> 16)
	x[2] = uint8(length >> 8)
	x[3] = uint8(length)

	x[4] = uint8(len(m.certificateTypes))

	copy(x[5:], m.certificateTypes)
	y := x[5+len(m.certificateTypes):]

	if m.hasSignatureAlgorithm {
		n := len(m.supportedSignatureAlgorithms) * 2
		y[0] = uint8(n >> 8)
		y[1] = uint8(n)
		y = y[2:]
		for _, sigAlgo := range m.supportedSignatureAlgorithms {
			y[0] = uint8(sigAlgo >> 8)
			y[1] = uint8(sigAlgo)
			y = y[2:]
		}
	}

	y[0] = uint8(casLength >> 8)
	y[1] = uint8(casLength)
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

	if m.hasSignatureAlgorithm {
		if len(data) < 2 {
			return false
		}
		sigAndHashLen := uint16(data[0])<<8 | uint16(data[1])
		data = data[2:]
		if sigAndHashLen&1 != 0 {
			return false
		}
		if len(data) < int(sigAndHashLen) {
			return false
		}
		numSigAlgos := sigAndHashLen / 2
		m.supportedSignatureAlgorithms = make([]SignatureScheme, numSigAlgos)
		for i := range m.supportedSignatureAlgorithms {
			m.supportedSignatureAlgorithms[i] = SignatureScheme(data[0])<<8 | SignatureScheme(data[1])
			data = data[2:]
		}
	}

	if len(data) < 2 {
		return false
	}
	casLength := uint16(data[0])<<8 | uint16(data[1])
	data = data[2:]
	if len(data) < int(casLength) {
		return false
	}
	cas := make([]byte, casLength)
	copy(cas, data)
	data = data[casLength:]

	m.certificateAuthorities = nil
	for len(cas) > 0 {
		if len(cas) < 2 {
			return false
		}
		caLen := uint16(cas[0])<<8 | uint16(cas[1])
		cas = cas[2:]

		if len(cas) < int(caLen) {
			return false
		}

		m.certificateAuthorities = append(m.certificateAuthorities, cas[:caLen])
		cas = cas[caLen:]
	}

	return len(data) == 0
}

type certificateVerifyMsg struct {
	raw                   []byte
	hasSignatureAlgorithm bool // format change introduced in TLS 1.2
	signatureAlgorithm    SignatureScheme
	signature             []byte
}

func (m *certificateVerifyMsg) marshal() (x []byte) {
	if m.raw != nil {
		return m.raw
	}

	var b cryptobyte.Builder
	b.AddUint8(typeCertificateVerify)
	b.AddUint24LengthPrefixed(func(b *cryptobyte.Builder) {
		if m.hasSignatureAlgorithm {
			b.AddUint16(uint16(m.signatureAlgorithm))
		}
		b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
			b.AddBytes(m.signature)
		})
	})

	m.raw = b.BytesOrPanic()
	return m.raw
}

func (m *certificateVerifyMsg) unmarshal(data []byte) bool {
	m.raw = data
	s := cryptobyte.String(data)

	if !s.Skip(4) { // message type and uint24 length field
		return false
	}
	if m.hasSignatureAlgorithm {
		if !s.ReadUint16((*uint16)(&m.signatureAlgorithm)) {
			return false
		}
	}
	return readUint16LengthPrefixed(&s, &m.signature) && s.Empty()
}

type newSessionTicketMsg struct {
	raw    []byte
	ticket []byte
}

func (m *newSessionTicketMsg) marshal() (x []byte) {
	if m.raw != nil {
		return m.raw
	}

	// See RFC 5077, Section 3.3.
	ticketLen := len(m.ticket)
	length := 2 + 4 + ticketLen
	x = make([]byte, 4+length)
	x[0] = typeNewSessionTicket
	x[1] = uint8(length >> 16)
	x[2] = uint8(length >> 8)
	x[3] = uint8(length)
	x[8] = uint8(ticketLen >> 8)
	x[9] = uint8(ticketLen)
	copy(x[10:], m.ticket)

	m.raw = x

	return
}

func (m *newSessionTicketMsg) unmarshal(data []byte) bool {
	m.raw = data

	if len(data) < 10 {
		return false
	}

	length := uint32(data[1])<<16 | uint32(data[2])<<8 | uint32(data[3])
	if uint32(len(data))-4 != length {
		return false
	}

	ticketLen := int(data[8])<<8 + int(data[9])
	if len(data)-10 != ticketLen {
		return false
	}

	m.ticket = data[10:]

	return true
}

type helloRequestMsg struct {
}

func (*helloRequestMsg) marshal() []byte {
	return []byte{typeHelloRequest, 0, 0, 0}
}

func (*helloRequestMsg) unmarshal(data []byte) bool {
	return len(data) == 4
}
