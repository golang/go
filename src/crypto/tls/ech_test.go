// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bytes"
	"crypto/ecdh"
	"crypto/hpke"
	"encoding/hex"
	"errors"
	"strings"
	"testing"

	"golang.org/x/crypto/cryptobyte"
)

func TestParseECHExt(t *testing.T) {
	for _, tc := range []struct {
		name, encoded string
		wantErr       error
	}{
		{"empty", "", errMalformedECHExt},
		{"inner", "01", nil},
		{"innerTrailingData", "0100", errMalformedECHExt},
		{"unknownType", "02", errInvalidECHExt},
		{"outer", "000001000107000201020003030405", nil},
		{"outerEmptyEnc", "00000100010700000003030405", nil},
		{"outerTrailingByte", "00000100010700020102000303040500", errMalformedECHExt},
		{"outerTrailingBytes", "00000100010700020102000303040500ff", errMalformedECHExt},
		{"outerEmptyEncTrailingData", "0000010001070000000303040500", errMalformedECHExt},
		{"truncatedEnc", "000001000107000201", errMalformedECHExt},
		{"truncatedPayload", "0000010001070002010200030304", errMalformedECHExt},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ext, err := hex.DecodeString(tc.encoded)
			if err != nil {
				t.Fatal(err)
			}
			echType, cs, id, enc, payload, err := parseECHExt(ext)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("parseECHExt: got %v, want %v", err, tc.wantErr)
			}
			if err != nil || echType == innerECHExt {
				return
			}
			if cs != (echCipher{KDFID: 1, AEADID: 1}) || id != 7 || !bytes.Equal(payload, []byte{3, 4, 5}) {
				t.Fatalf("unexpected outer extension: ciphersuite=%v id=%v payload=%x", cs, id, payload)
			}
			if tc.name == "outer" && !bytes.Equal(enc, []byte{1, 2}) || tc.name == "outerEmptyEnc" && len(enc) != 0 {
				t.Fatalf("unexpected encapsulated key: %x", enc)
			}
			// Parsed slices must not alias the extension on the wire.
			original := bytes.Clone(ext)
			clear(enc)
			clear(payload)
			if !bytes.Equal(ext, original) {
				t.Fatal("mutating parsed fields changed the raw extension")
			}
		})
	}
}

func TestECHOuterTrailingData(t *testing.T) {
	key, err := hpke.DHKEM(ecdh.X25519()).GenerateKey()
	if err != nil {
		t.Fatal(err)
	}
	privateKey, err := key.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	var b cryptobyte.Builder
	b.AddUint16(extensionEncryptedClientHello)
	b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
		b.AddUint8(7) // config_id
		b.AddUint16(key.KEM().ID())
		b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) { b.AddBytes(key.PublicKey().Bytes()) })
		b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
			b.AddUint16(hpke.HKDFSHA256().ID())
			b.AddUint16(hpke.AES128GCM().ID())
		})
		b.AddUint8(0) // maximum_name_length
		b.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) { b.AddBytes([]byte("public.example")) })
		b.AddUint16(0) // extensions
	})
	config := b.BytesOrPanic()
	for _, trailing := range []bool{false, true} {
		name := "valid"
		if trailing {
			name = "trailingData"
		}
		t.Run(name, func(t *testing.T) {
			enc, sender, err := hpke.NewSender(key.PublicKey(), hpke.HKDFSHA256(), hpke.AES128GCM(), append([]byte("tls ech\x00"), config...))
			if err != nil {
				t.Fatal(err)
			}
			inner := &clientHelloMsg{
				vers:                 VersionTLS12,
				random:               make([]byte, 32),
				serverName:           "secret.example",
				cipherSuites:         []uint16{TLS_AES_128_GCM_SHA256},
				compressionMethods:   []uint8{0},
				supportedVersions:    []uint16{VersionTLS13},
				encryptedClientHello: []byte{uint8(innerECHExt)},
			}
			encodedInner, err := encodeInnerClientHello(inner, 0)
			if err != nil {
				t.Fatal(err)
			}
			outer := inner.clone()
			outer.serverName = "public.example"
			setPayload := func(payload []byte) {
				t.Helper()
				ext, err := generateOuterECHExt(7, hpke.HKDFSHA256().ID(), hpke.AES128GCM().ID(), enc, payload)
				if err != nil {
					t.Fatal(err)
				}
				if trailing {
					ext = append(ext, 0xff)
				}
				outer.encryptedClientHello = ext
			}
			// Seal against the outer hello including the suffix. Merely appending
			// a byte after sealing would instead cause HPKE decryption to fail.
			setPayload(make([]byte, len(encodedInner)+16)) // AES-GCM tag length
			aad, err := outer.marshal()
			if err != nil {
				t.Fatal(err)
			}
			payload, err := sender.Seal(aad[4:], encodedInner)
			if err != nil {
				t.Fatal(err)
			}
			setPayload(payload)
			wire, err := outer.marshal()
			if err != nil {
				t.Fatal(err)
			}
			outer = new(clientHelloMsg)
			if !outer.unmarshal(wire) {
				t.Fatal("failed to unmarshal outer ClientHello")
			}
			server := Server(&discardConn{}, testConfigServer())
			got, _, err := server.processECHClientHello(outer, []EncryptedClientHelloKey{{Config: config, PrivateKey: privateKey}})
			if trailing {
				if !errors.Is(err, errInvalidECHExt) || !errors.Is(server.out.err, alertDecodeError) || server.echAccepted {
					t.Fatalf("trailing data: error=%v alert=%v ECHAccepted=%v", err, server.out.err, server.echAccepted)
				}
			} else if err != nil || !server.echAccepted || got.serverName != inner.serverName {
				t.Fatalf("valid ECH: hello=%v error=%v ECHAccepted=%v", got, err, server.echAccepted)
			}
		})
	}
}

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
