// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bytes"
	"crypto/ecdh"
	"crypto/hpke"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"io"
	"strings"
	"testing"

	"golang.org/x/crypto/cryptobyte"
)

func TestDecodeECHConfigLists(t *testing.T) {
	for _, tc := range []struct {
		list       string
		numConfigs int
	}{
		{"0045fe0d0041590020002092a01233db2218518ccbbbbc24df20686af417b37388de6460e94011974777090004000100010012636c6f7564666c6172652d6563682e636f6d0000", 1},
		{"0105badd00050504030201fe0d0066000010004104e62b69e2bf659f97be2f1e0d948a4cd5976bb7a91e0d46fbdda9a91e9ddcba5a01e7d697a80a18f9c3c4a31e56e27c8348db161a1cf51d7ef1942d4bcf7222c1000c000100010001000200010003400e7075626c69632e6578616d706c650000fe0d003d00002000207d661615730214aeee70533366f36a609ead65c0c208e62322346ab5bcd8de1c000411112222400e7075626c69632e6578616d706c650000fe0d004d000020002085bd6a03277c25427b52e269e0c77a8eb524ba1eb3d2f132662d4b0ac6cb7357000c000100010001000200010003400e7075626c69632e6578616d706c650008aaaa000474657374", 3},
	} {
		b, err := hex.DecodeString(tc.list)
		if err != nil {
			t.Fatal(err)
		}
		configs, err := parseECHConfigList(b)
		if err != nil {
			t.Fatal(err)
		}
		if len(configs) != tc.numConfigs {
			t.Fatalf("unexpected number of configs parsed: got %d want %d", len(configs), tc.numConfigs)
		}
	}

}

func TestSkipBadConfigs(t *testing.T) {
	b, err := hex.DecodeString("00c8badd00050504030201fe0d0029006666000401020304000c000100010001000200010003400e7075626c69632e6578616d706c650000fe0d003d000020002072e8a23b7aef67832bcc89d652e3870a60f88ca684ec65d6eace6b61f136064c000411112222400e7075626c69632e6578616d706c650000fe0d004d00002000200ce95810a81d8023f41e83679bc92701b2acd46c75869f95c72bc61c6b12297c000c000100010001000200010003400e7075626c69632e6578616d706c650008aaaa000474657374")
	if err != nil {
		t.Fatal(err)
	}
	configs, err := parseECHConfigList(b)
	if err != nil {
		t.Fatal(err)
	}
	config, _, _, _ := pickECHConfig(configs)
	if config != nil {
		t.Fatal("pickECHConfig picked an invalid config")
	}
}

func TestPickECHConfigWithInvalidAEADID(t *testing.T) {
	b, err := hex.DecodeString("0045fe0d0041590020002092a01233db2218518ccbbbbc24df20686af417b37388de6460e94011974777090004000100010012636c6f7564666c6172652d6563682e636f6d0000")
	if err != nil {
		t.Fatal(err)
	}
	buf := bytes.Replace(b, []byte{0x00, 0x01, 0x00, 0x01}, []byte{0x00, 0x01, 0xFF, 0xFF}, 1)
	configs, err := parseECHConfigList(buf)
	if err != nil {
		t.Fatal(err)
	}
	if config, _, _, _ := pickECHConfig(configs); config != nil {
		t.Fatalf("got %v, want nil", config)
	}
}

func TestECHPadding(t *testing.T) {
	const maxNameLength = 64
	for _, tc := range []struct {
		name       string
		serverName string
	}{
		{"Short", "a.test"},
		{"Medium", strings.Repeat("a", 30) + ".test"},
		{"MaxLength", strings.Repeat("a", maxNameLength) + ".test"},
		{"NoSNI", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			inner := &clientHelloMsg{
				vers:               VersionTLS13,
				random:             make([]byte, 32),
				serverName:         tc.serverName,
				cipherSuites:       []uint16{TLS_AES_128_GCM_SHA256},
				compressionMethods: []uint8{0},
				supportedVersions:  []uint16{VersionTLS13},
			}
			encoded, err := encodeInnerClientHello(inner, maxNameLength)
			if err != nil {
				t.Fatal(err)
			}
			if len(encoded)%32 != 0 {
				t.Errorf("got %d, want multiple of 32", len(encoded))
			}
		})
	}

	t.Run("SetSizeReduction", func(t *testing.T) {
		sizes := make(map[int]struct{})
		for sniLen := 1; sniLen <= maxNameLength; sniLen++ {
			inner := &clientHelloMsg{
				vers:               VersionTLS13,
				random:             make([]byte, 32),
				serverName:         strings.Repeat("a", sniLen) + ".test",
				cipherSuites:       []uint16{TLS_AES_128_GCM_SHA256},
				compressionMethods: []uint8{0},
				supportedVersions:  []uint16{VersionTLS13},
			}
			encoded, err := encodeInnerClientHello(inner, maxNameLength)
			if err != nil {
				t.Fatal(err)
			}
			sizes[len(encoded)] = struct{}{}
		}
		if len(sizes) > 4 {
			t.Errorf("got %d distinct encoded sizes for SNI lengths 1..%d, want <= 4", len(sizes), maxNameLength)
		}
	})
}

func TestDecodeInnerClientHelloOuterExtensions(t *testing.T) {
	outer := &clientHelloMsg{
		vers:                 VersionTLS12,
		random:               make([]byte, 32),
		cipherSuites:         []uint16{TLS_AES_128_GCM_SHA256},
		compressionMethods:   []uint8{compressionNone},
		ocspStapling:         true,
		supportedCurves:      []CurveID{CurveP256},
		encryptedClientHello: []byte{byte(innerECHExt)},
	}
	outer.original = mustMarshal(t, outer)

	encodeInner := func(buildExts func(*cryptobyte.Builder)) []byte {
		var b cryptobyte.Builder
		b.AddUint16(VersionTLS12)
		b.AddBytes(make([]byte, 32))
		b.AddUint8(0)
		b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
			b.AddUint16(TLS_AES_128_GCM_SHA256)
		})
		b.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) {
			b.AddUint8(compressionNone)
		})
		b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
			buildExts(b)
			b.AddUint16(extensionEncryptedClientHello)
			b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
				b.AddUint8(uint8(innerECHExt))
			})
			b.AddUint16(extensionSupportedVersions)
			b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
				b.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) {
					b.AddUint16(VersionTLS13)
				})
			})
		})
		return b.BytesOrPanic()
	}

	outerExts := func(b *cryptobyte.Builder, extTypes ...uint16) {
		b.AddUint16(extensionECHOuterExtensions)
		b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
			b.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) {
				for _, extType := range extTypes {
					b.AddUint16(extType)
				}
			})
		})
	}

	for _, tc := range []struct {
		name      string
		buildExts func(*cryptobyte.Builder)
		wantErr   string
	}{
		{
			name: "valid order",
			buildExts: func(b *cryptobyte.Builder) {
				outerExts(b, extensionStatusRequest, extensionSupportedCurves)
			},
		},
		{
			name: "duplicate reference",
			buildExts: func(b *cryptobyte.Builder) {
				outerExts(b, extensionStatusRequest, extensionStatusRequest)
			},
			wantErr: "tls: invalid outer extensions",
		},
		{
			name: "references encrypted_client_hello",
			buildExts: func(b *cryptobyte.Builder) {
				outerExts(b, extensionEncryptedClientHello)
			},
			wantErr: "tls: invalid outer extensions",
		},
		{
			name: "references ech_outer_extensions",
			buildExts: func(b *cryptobyte.Builder) {
				outerExts(b, extensionECHOuterExtensions)
			},
			wantErr: "tls: invalid outer extensions",
		},
		{
			name: "multiple ech_outer_extensions",
			buildExts: func(b *cryptobyte.Builder) {
				outerExts(b, extensionStatusRequest)
				outerExts(b, extensionSupportedCurves)
			},
			wantErr: "tls: invalid outer extensions",
		},
		{
			name: "odd-length reference list",
			buildExts: func(b *cryptobyte.Builder) {
				b.AddUint16(extensionECHOuterExtensions)
				b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
					b.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) {
						b.AddUint8(0)
					})
				})
			},
			wantErr: "tls: invalid inner client hello",
		},
		{
			name: "empty reference list",
			buildExts: func(b *cryptobyte.Builder) {
				b.AddUint16(extensionECHOuterExtensions)
				b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
					b.AddUint8(0)
				})
			},
			wantErr: "tls: invalid inner client hello",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			encoded := encodeInner(tc.buildExts)
			_, err := decodeInnerClientHello(outer, encoded)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("decodeInnerClientHello returned %v, want nil", err)
				}
			} else if err == nil || err.Error() != tc.wantErr {
				t.Fatalf("decodeInnerClientHello returned %v, want %q", err, tc.wantErr)
			}
		})
	}
}

func TestDecodeECHConfigListOverflow(t *testing.T) {
	// Craft a 65538-byte payload with an outer length header of 0 (bytes 0-1)
	// and an inner length of 65532 (bytes 4-5). Previously, both lengths were
	// handled as uint16 and would overflow:
	// 1. uint16(65538-2) becomes 0, causing it to erroneously match the
	// declared outer length header of 0.
	// 2. uint16(65532+4) becomes 0, making it so that when we advance past the
	// inner ECHConfig, we would only advance the buffer by 0 bytes, causing an
	// infinite loop.
	payload := make([]byte, 65538)
	payload[4] = 0xFF
	payload[5] = 0xFC
	if _, err := parseECHConfigList(payload); !errors.Is(err, errMalformedECHConfigList) {
		t.Fatalf("got %v when parsing a malformed ECHConfigList; want %v", err, errMalformedECHConfigList)
	}
}

func TestPickECHConfigWithIPPublicName(t *testing.T) {
	b, err := hex.DecodeString("0045fe0d0041590020002092a01233db2218518ccbbbbc24df20686af417b37388de6460e94011974777090004000100010012636c6f7564666c6172652d6563682e636f6d0000")
	if err != nil {
		t.Fatal(err)
	}
	configs, err := parseECHConfigList(b)
	if err != nil {
		t.Fatal(err)
	}
	valid := configs[0]
	for _, name := range []string{
		"127.0.0.1",
		"::1",
	} {
		t.Run(name, func(t *testing.T) {
			invalid := valid
			invalid.PublicName = []byte(name)
			if config, _, _, _ := pickECHConfig([]echConfig{invalid}); config != nil {
				t.Error("selected an ECH config with an IP public name")
			}
			config, _, _, _ := pickECHConfig([]echConfig{invalid, valid})
			if config == nil || !bytes.Equal(config.PublicName, valid.PublicName) {
				t.Error("did not select the subsequent config with a DNS public name")
			}
		})
	}
}

func TestECHServerCipherSuite(t *testing.T) {
	key, err := ecdh.X25519().GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	publicKey, err := hpke.NewDHKEMPublicKey(key.PublicKey())
	if err != nil {
		t.Fatal(err)
	}
	makeConfig := func(suite echCipher) []byte {
		var b cryptobyte.Builder
		b.AddUint16(extensionEncryptedClientHello)
		b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
			b.AddUint8(1)
			b.AddUint16(0x0020)
			b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) { b.AddBytes(key.PublicKey().Bytes()) })
			b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) { b.AddUint16(suite.KDFID); b.AddUint16(suite.AEADID) })
			b.AddUint8(0)
			b.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) { b.AddBytes([]byte("public.example")) })
			b.AddUint16(0)
		})
		raw, err := b.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		return raw
	}
	for _, tc := range []struct {
		name            string
		kdf             hpke.KDF
		aead            hpke.AEAD
		advertised      bool
		precedingConfig bool
	}{
		{"advertised", hpke.HKDFSHA256(), hpke.AES128GCM(), true, false},
		{"unadvertised AEAD", hpke.HKDFSHA256(), hpke.AES256GCM(), false, false},
		{"unadvertised KDF", hpke.HKDFSHA384(), hpke.AES128GCM(), false, false},
		{"later matching config", hpke.HKDFSHA256(), hpke.AES256GCM(), true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := makeConfig(echCipher{1, 1})
			if tc.advertised {
				raw = makeConfig(echCipher{tc.kdf.ID(), tc.aead.ID()})
			}
			_, config, err := parseECHConfig(raw)
			if err != nil {
				t.Fatal(err)
			}
			enc, sender, err := hpke.NewSender(publicKey, tc.kdf, tc.aead, append([]byte("tls ech\x00"), raw...))
			if err != nil {
				t.Fatal(err)
			}
			inner := &clientHelloMsg{vers: VersionTLS12, random: make([]byte, 32), cipherSuites: []uint16{TLS_AES_128_GCM_SHA256}, compressionMethods: []uint8{0}, supportedVersions: []uint16{VersionTLS13}, serverName: "secret.example", encryptedClientHello: []byte{uint8(innerECHExt)}}
			outer := *inner
			outer.serverName = "public.example"
			ctx := &echClientContext{config: &config, kdfID: tc.kdf.ID(), aeadID: tc.aead.ID(), encapsulatedKey: enc, hpkeContext: sender}
			if err := computeAndUpdateOuterECHExtension(&outer, inner, ctx, true); err != nil {
				t.Fatal(err)
			}
			outer.original, err = outer.marshal()
			if err != nil {
				t.Fatal(err)
			}
			keys := []EncryptedClientHelloKey{{Config: raw, PrivateKey: key.Bytes()}}
			if tc.precedingConfig {
				keys = append([]EncryptedClientHelloKey{{Config: makeConfig(echCipher{1, 1}), PrivateKey: key.Bytes()}}, keys...)
			}
			c := &Conn{}
			hello, server, err := c.processECHClientHello(&outer, keys)
			if err != nil {
				t.Fatal(err)
			}
			if c.echAccepted != tc.advertised || (server != nil) != tc.advertised {
				t.Fatalf("ECH accepted = %v, want %v", c.echAccepted, tc.advertised)
			}
			if tc.advertised {
				if hello.serverName != inner.serverName {
					t.Fatalf("inner server name = %q", hello.serverName)
				}
			} else if hello != &outer {
				t.Fatal("rejection did not preserve outer ClientHello")
			}
		})
	}
}

// TestECHOuterHelloOmitsPSK verifies computeAndUpdateOuterECHExtension strips
// the inner hello's PSK offer from the outer hello.
func TestECHOuterHelloOmitsPSK(t *testing.T) {
	// Generate the ECH config in-test so the private key is known and the
	// sealed payload can be decrypted below.
	echKey, err := ecdh.X25519().GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	configBuilder := cryptobyte.NewBuilder(nil)
	configBuilder.AddUint16(extensionEncryptedClientHello)
	configBuilder.AddUint16LengthPrefixed(func(builder *cryptobyte.Builder) {
		builder.AddUint8(1) // config_id
		builder.AddUint16(0x0020 /* DHKEM(X25519, HKDF-SHA256) */)
		builder.AddUint16LengthPrefixed(func(builder *cryptobyte.Builder) {
			builder.AddBytes(echKey.PublicKey().Bytes())
		})
		builder.AddUint16LengthPrefixed(func(builder *cryptobyte.Builder) {
			builder.AddUint16(0x0001 /* HKDF-SHA256 */)
			builder.AddUint16(0x0001 /* AES-128-GCM */)
		})
		builder.AddUint8(32) // maximum_name_length
		builder.AddUint8LengthPrefixed(func(builder *cryptobyte.Builder) {
			builder.AddBytes([]byte("public.example"))
		})
		builder.AddUint16(0) // extensions
	})
	echConfigBytes := configBuilder.BytesOrPanic()
	listBuilder := cryptobyte.NewBuilder(nil)
	listBuilder.AddUint16LengthPrefixed(func(builder *cryptobyte.Builder) {
		builder.AddBytes(echConfigBytes)
	})

	configs, err := parseECHConfigList(listBuilder.BytesOrPanic())
	if err != nil {
		t.Fatal(err)
	}
	config, echPK, kdf, aead := pickECHConfig(configs)
	if config == nil {
		t.Fatal("pickECHConfig returned no config")
	}
	ech := &echClientContext{config: config, kdfID: kdf.ID(), aeadID: aead.ID()}
	info := append([]byte("tls ech\x00"), config.raw...)
	ech.encapsulatedKey, ech.hpkeContext, err = hpke.NewSender(echPK, kdf, aead, info)
	if err != nil {
		t.Fatal(err)
	}

	inner := &clientHelloMsg{
		vers:                 VersionTLS13,
		random:               make([]byte, 32),
		sessionId:            make([]byte, 32),
		serverName:           "secret.example",
		cipherSuites:         []uint16{TLS_AES_128_GCM_SHA256},
		compressionMethods:   []uint8{0},
		supportedVersions:    []uint16{VersionTLS13},
		encryptedClientHello: []byte{byte(innerECHExt)},
		pskIdentities:        []pskIdentity{{label: []byte("ticket")}},
		pskBinders:           [][]byte{make([]byte, 32)},
	}
	outer := inner.clone()
	outer.serverName = string(config.PublicName)

	if err := computeAndUpdateOuterECHExtension(outer, inner, ech, true); err != nil {
		t.Fatal(err)
	}

	if outer.pskIdentities != nil || outer.pskBinders != nil {
		t.Error("outer hello retains PSK state after computeAndUpdateOuterECHExtension")
	}
	marshaled, err := outer.marshal()
	if err != nil {
		t.Fatal(err)
	}
	var parsed clientHelloMsg
	if !parsed.unmarshal(marshaled) {
		t.Fatal("failed to unmarshal outer hello")
	}
	if len(parsed.pskIdentities) != 0 || len(parsed.pskBinders) != 0 {
		t.Error("marshaled outer hello contains a pre_shared_key extension")
	}
	if len(inner.pskIdentities) != 1 || len(inner.pskBinders) != 1 {
		t.Error("inner hello lost its PSK offer")
	}

	// Stripping the PSK from the outer hello must not leak into the inner
	// hello's encoding. The sealed inner is the only place the offer is
	// carried, so losing it there silently disables resumption. Decrypt the
	// sealed payload as the server would and check the offer survived.
	echType, echCipherSuite, _, encap, payload, err := parseECHExt(parsed.encryptedClientHello)
	if err != nil {
		t.Fatal(err)
	}
	if echType != outerECHExt {
		t.Fatalf("ech extension type: got %v, want %v", echType, outerECHExt)
	}
	kem, err := hpke.NewKEM(config.KemID)
	if err != nil {
		t.Fatal(err)
	}
	echPriv, err := kem.NewPrivateKey(echKey.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	recipientKDF, err := hpke.NewKDF(echCipherSuite.KDFID)
	if err != nil {
		t.Fatal(err)
	}
	recipientAEAD, err := hpke.NewAEAD(echCipherSuite.AEADID)
	if err != nil {
		t.Fatal(err)
	}
	recipient, err := hpke.NewRecipient(encap, echPriv, recipientKDF, recipientAEAD, info)
	if err != nil {
		t.Fatal(err)
	}
	encodedInner, err := decryptECHPayload(recipient, parsed.original, payload)
	if err != nil {
		t.Fatal(err)
	}
	decodedInner, err := decodeInnerClientHello(&parsed, encodedInner)
	if err != nil {
		t.Fatal(err)
	}
	if len(decodedInner.pskIdentities) != 1 || len(decodedInner.pskBinders) != 1 {
		t.Error("decoded inner hello lost its pre_shared_key offer")
	}
}

// TestGreaseECHGodebugOptOut verifies that GODEBUG=tlsechgrease=0 fully
// disables sending a GREASE'd encrypted_client_hello extension.
func TestGreaseECHGodebugOptOut(t *testing.T) {
	newConn := func() *Conn {
		return &Conn{config: &Config{
			ServerName:         "test.example",
			MinVersion:         VersionTLS13,
			InsecureSkipVerify: true,
		}}
	}

	t.Run("default-on", func(t *testing.T) {
		t.Setenv("GODEBUG", "")
		conn := newConn()
		hello, _, ech, err := conn.makeClientHello()
		if err != nil {
			t.Fatal(err)
		}
		if ech != nil {
			t.Errorf("ech context: got %#v, want nil for GREASE", ech)
		}
		if !conn.echGrease {
			t.Errorf("conn.echGrease: got false, want true")
		}
		if len(hello.encryptedClientHello) == 0 {
			t.Errorf("encryptedClientHello empty, want a GREASE extension")
		}
	})

	t.Run("opt-out", func(t *testing.T) {
		t.Setenv("GODEBUG", "tlsechgrease=0")
		conn := newConn()
		hello, _, ech, err := conn.makeClientHello()
		if err != nil {
			t.Fatal(err)
		}
		if ech != nil {
			t.Errorf("ech context: got %#v, want nil under tlsechgrease=0", ech)
		}
		if conn.echGrease {
			t.Errorf("conn.echGrease: got true, want false under tlsechgrease=0")
		}
		if len(hello.encryptedClientHello) != 0 {
			t.Errorf("encryptedClientHello: got %d bytes, want 0", len(hello.encryptedClientHello))
		}
	})
}

// TestGreaseECHHRRMalformedExtension verifies a GREASE'ing client MUST
// abort with a decode_error alert when the server's HRR includes a
// malformed encrypted_client_hello extension.
func TestGreaseECHHRRMalformedExtension(t *testing.T) {
	c, s := localPipe(t)
	errChan := make(chan error, 1)

	go func() {
		errChan <- Client(c, &Config{
			ServerName:         "test.example",
			MinVersion:         VersionTLS13,
			InsecureSkipVerify: true,
		}).Handshake()
	}()

	// Read the ClientHello record so we can echo session_id back in the HRR.
	var header [5]byte
	if _, err := io.ReadFull(s, header[:]); err != nil {
		t.Fatal(err)
	}
	record := make([]byte, int(header[3])<<8|int(header[4]))
	if _, err := io.ReadFull(s, record); err != nil {
		t.Fatal(err)
	}
	var clientHello clientHelloMsg
	if !clientHello.unmarshal(record) {
		t.Fatal("failed to parse ClientHello")
	}
	if len(clientHello.supportedCurves) == 0 {
		t.Fatal("ClientHello has no supportedCurves; cannot build HRR")
	}

	// Drain any subsequent writes from the client so they don't block.
	go io.Copy(io.Discard, s)

	// Build a HelloRetryRequest carrying a 7-byte ECH extension (length != 8
	// per RFC 9849 §6.2).
	hrr := &serverHelloMsg{
		vers:                 VersionTLS12,
		random:               helloRetryRequestRandom,
		sessionId:            clientHello.sessionId,
		cipherSuite:          TLS_AES_128_GCM_SHA256,
		compressionMethod:    0,
		supportedVersion:     VersionTLS13,
		encryptedClientHello: bytes.Repeat([]byte{0}, 7),
		selectedGroup:        clientHello.supportedCurves[0],
	}
	hrrBytes := mustMarshal(t, hrr)
	if _, err := s.Write([]byte{
		byte(recordTypeHandshake),
		byte(VersionTLS12 >> 8),
		byte(VersionTLS12 & 0xff),
		byte(len(hrrBytes) >> 8),
		byte(len(hrrBytes)),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Write(hrrBytes); err != nil {
		t.Fatal(err)
	}

	err := <-errChan
	if err == nil {
		t.Fatal("handshake unexpectedly succeeded; want error on malformed HRR ECH ext")
	}
	want := "malformed encrypted client hello extension"
	if !strings.Contains(err.Error(), want) {
		t.Fatalf("handshake error: got %q, want %q", err, want)
	}
}

func TestECHEmptyExtensionInTLS12ServerHello(t *testing.T) {
	// A TLS 1.2 ServerHello selecting ECDHE-ECDSA-AES128-GCM-SHA256
	// with a single, empty encrypted_client_hello extension.
	hello, err := hex.DecodeString("0200002c0303" + strings.Repeat("00", 32) + "00c02b000004fe0d0000")
	if err != nil {
		t.Fatal(err)
	}
	clientConn, peer := localPipe(t)
	peer.Close()
	defer clientConn.Close()
	c := Client(&discardConn{clientConn}, testConfigClient())
	c.handBuf().Write(hello)
	if err := c.Handshake(); err == nil || !errors.Is(c.out.err, alertUnsupportedExtension) {
		t.Fatalf("Handshake returned %v with alert %v, want unsupported_extension", err, c.out.err)
	}
}

func TestGreaseECHInFinalServerHello(t *testing.T) {
	t.Setenv("GODEBUG", "tlsechgrease=1")
	c := Client(&discardConn{}, testConfigClient())
	hello, _, _, err := c.makeClientHello()
	if err != nil {
		t.Fatal(err)
	}
	hs := clientHandshakeStateTLS13{c: c, hello: hello, serverHello: &serverHelloMsg{
		vers:                 VersionTLS12,
		supportedVersion:     VersionTLS13,
		sessionId:            hello.sessionId,
		random:               make([]byte, 32),
		cipherSuite:          TLS_AES_128_GCM_SHA256,
		encryptedClientHello: make([]byte, 8),
	}}
	if err := hs.checkServerHelloOrHRR(); err == nil || !errors.Is(c.out.err, alertIllegalParameter) {
		t.Fatalf("checkServerHelloOrHRR returned %v with alert %v, want illegal_parameter", err, c.out.err)
	}
}

func TestUnsolicitedECHRetryConfigs(t *testing.T) {
	t.Setenv("GODEBUG", "tlsechgrease=0")
	c := Client(&discardConn{}, testConfigClient())
	hello, _, _, err := c.makeClientHello()
	if err != nil {
		t.Fatal(err)
	}
	// An ECHConfigList with one config of an unknown version.
	c.handBuf().Write(mustMarshal(t, &encryptedExtensionsMsg{echRetryConfigs: []byte{0, 4, 0xba, 0xdd, 0, 0}}))
	hs := clientHandshakeStateTLS13{c: c, hello: hello}
	if err := hs.readServerParameters(); err == nil || !errors.Is(c.out.err, alertUnsupportedExtension) {
		t.Fatalf("readServerParameters returned %v with alert %v, want unsupported_extension", err, c.out.err)
	}
}
