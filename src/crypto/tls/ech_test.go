// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bytes"
	"encoding/hex"
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
