// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bytes"
	"encoding/hex"
	"hash"
	"strings"
	"testing"
	"unicode"
)

// This file contains tests derived from draft-ietf-tls-tls13-vectors-07.

func parseVector(v string) []byte {
	v = strings.Map(func(c rune) rune {
		if unicode.IsSpace(c) {
			return -1
		}
		return c
	}, v)
	parts := strings.Split(v, ":")
	v = parts[len(parts)-1]
	res, err := hex.DecodeString(v)
	if err != nil {
		panic(err)
	}
	return res
}

func TestDeriveSecret(t *testing.T) {
	chTranscript := cipherSuitesTLS13[0].hash.New()
	chTranscript.Write(parseVector(`
	payload (512 octets):  01 00 01 fc 03 03 1b c3 ce b6 bb e3 9c ff
	93 83 55 b5 a5 0a db 6d b2 1b 7a 6a f6 49 d7 b4 bc 41 9d 78 76
	48 7d 95 00 00 06 13 01 13 03 13 02 01 00 01 cd 00 00 00 0b 00
	09 00 00 06 73 65 72 76 65 72 ff 01 00 01 00 00 0a 00 14 00 12
	00 1d 00 17 00 18 00 19 01 00 01 01 01 02 01 03 01 04 00 33 00
	26 00 24 00 1d 00 20 e4 ff b6 8a c0 5f 8d 96 c9 9d a2 66 98 34
	6c 6b e1 64 82 ba dd da fe 05 1a 66 b4 f1 8d 66 8f 0b 00 2a 00
	00 00 2b 00 03 02 03 04 00 0d 00 20 00 1e 04 03 05 03 06 03 02
	03 08 04 08 05 08 06 04 01 05 01 06 01 02 01 04 02 05 02 06 02
	02 02 00 2d 00 02 01 01 00 1c 00 02 40 01 00 15 00 57 00 00 00
	00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
	00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
	00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
	00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
	00 29 00 dd 00 b8 00 b2 2c 03 5d 82 93 59 ee 5f f7 af 4e c9 00
	00 00 00 26 2a 64 94 dc 48 6d 2c 8a 34 cb 33 fa 90 bf 1b 00 70
	ad 3c 49 88 83 c9 36 7c 09 a2 be 78 5a bc 55 cd 22 60 97 a3 a9
	82 11 72 83 f8 2a 03 a1 43 ef d3 ff 5d d3 6d 64 e8 61 be 7f d6
	1d 28 27 db 27 9c ce 14 50 77 d4 54 a3 66 4d 4e 6d a4 d2 9e e0
	37 25 a6 a4 da fc d0 fc 67 d2 ae a7 05 29 51 3e 3d a2 67 7f a5
	90 6c 5b 3f 7d 8f 92 f2 28 bd a4 0d da 72 14 70 f9 fb f2 97 b5
	ae a6 17 64 6f ac 5c 03 27 2e 97 07 27 c6 21 a7 91 41 ef 5f 7d
	e6 50 5e 5b fb c3 88 e9 33 43 69 40 93 93 4a e4 d3 57 fa d6 aa
	cb 00 21 20 3a dd 4f b2 d8 fd f8 22 a0 ca 3c f7 67 8e f5 e8 8d
	ae 99 01 41 c5 92 4d 57 bb 6f a3 1b 9e 5f 9d`))

	type args struct {
		secret     []byte
		label      string
		transcript hash.Hash
	}
	tests := []struct {
		name string
		args args
		want []byte
	}{
		{
			`derive secret for handshake "tls13 derived"`,
			args{
				parseVector(`PRK (32 octets):  33 ad 0a 1c 60 7e c0 3b 09 e6 cd 98 93 68 0c e2
				10 ad f3 00 aa 1f 26 60 e1 b2 2e 10 f1 70 f9 2a`),
				"derived",
				nil,
			},
			parseVector(`expanded (32 octets):  6f 26 15 a1 08 c7 02 c5 67 8f 54 fc 9d ba
			b6 97 16 c0 76 18 9c 48 25 0c eb ea c3 57 6c 36 11 ba`),
		},
		{
			`derive secret "tls13 c e traffic"`,
			args{
				parseVector(`PRK (32 octets):  9b 21 88 e9 b2 fc 6d 64 d7 1d c3 29 90 0e 20 bb
				41 91 50 00 f6 78 aa 83 9c bb 79 7c b7 d8 33 2c`),
				"c e traffic",
				chTranscript,
			},
			parseVector(`expanded (32 octets):  3f bb e6 a6 0d eb 66 c3 0a 32 79 5a ba 0e
			ff 7e aa 10 10 55 86 e7 be 5c 09 67 8d 63 b6 ca ab 62`),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := cipherSuitesTLS13[0]
			if got := c.deriveSecret(tt.args.secret, tt.args.label, tt.args.transcript); !bytes.Equal(got, tt.want) {
				t.Errorf("cipherSuiteTLS13.deriveSecret() = % x, want % x", got, tt.want)
			}
		})
	}
}

func TestTrafficKey(t *testing.T) {
	trafficSecret := parseVector(
		`PRK (32 octets):  b6 7b 7d 69 0c c1 6c 4e 75 e5 42 13 cb 2d 37 b4
		e9 c9 12 bc de d9 10 5d 42 be fd 59 d3 91 ad 38`)
	wantKey := parseVector(
		`key expanded (16 octets):  3f ce 51 60 09 c2 17 27 d0 f2 e4 e8 6e
		e4 03 bc`)
	wantIV := parseVector(
		`iv expanded (12 octets):  5d 31 3e b2 67 12 76 ee 13 00 0b 30`)

	c := cipherSuitesTLS13[0]
	gotKey, gotIV := c.trafficKey(trafficSecret)
	if !bytes.Equal(gotKey, wantKey) {
		t.Errorf("cipherSuiteTLS13.trafficKey() gotKey = % x, want % x", gotKey, wantKey)
	}
	if !bytes.Equal(gotIV, wantIV) {
		t.Errorf("cipherSuiteTLS13.trafficKey() gotIV = % x, want % x", gotIV, wantIV)
	}
}

func TestExtract(t *testing.T) {
	type args struct {
		newSecret     []byte
		currentSecret []byte
	}
	tests := []struct {
		name string
		args args
		want []byte
	}{
		{
			`extract secret "early"`,
			args{
				nil,
				nil,
			},
			parseVector(`secret (32 octets):  33 ad 0a 1c 60 7e c0 3b 09 e6 cd 98 93 68 0c
			e2 10 ad f3 00 aa 1f 26 60 e1 b2 2e 10 f1 70 f9 2a`),
		},
		{
			`extract secret "master"`,
			args{
				nil,
				parseVector(`salt (32 octets):  43 de 77 e0 c7 77 13 85 9a 94 4d b9 db 25 90 b5
				31 90 a6 5b 3e e2 e4 f1 2d d7 a0 bb 7c e2 54 b4`),
			},
			parseVector(`secret (32 octets):  18 df 06 84 3d 13 a0 8b f2 a4 49 84 4c 5f 8a
			47 80 01 bc 4d 4c 62 79 84 d5 a4 1d a8 d0 40 29 19`),
		},
		{
			`extract secret "handshake"`,
			args{
				parseVector(`IKM (32 octets):  8b d4 05 4f b5 5b 9d 63 fd fb ac f9 f0 4b 9f 0d
				35 e6 d6 3f 53 75 63 ef d4 62 72 90 0f 89 49 2d`),
				parseVector(`salt (32 octets):  6f 26 15 a1 08 c7 02 c5 67 8f 54 fc 9d ba b6 97
				16 c0 76 18 9c 48 25 0c eb ea c3 57 6c 36 11 ba`),
			},
			parseVector(`secret (32 octets):  1d c8 26 e9 36 06 aa 6f dc 0a ad c1 2f 74 1b
			01 04 6a a6 b9 9f 69 1e d2 21 a9 f0 ca 04 3f be ac`),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := cipherSuitesTLS13[0]
			if got := c.extract(tt.args.newSecret, tt.args.currentSecret); !bytes.Equal(got, tt.want) {
				t.Errorf("cipherSuiteTLS13.extract() = % x, want % x", got, tt.want)
			}
		})
	}
}
