// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bytes"
	"crypto/internal/fips140/mlkem"
	"crypto/internal/fips140/tls13"
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"
	"unicode"
)

func TestACVPVectors(t *testing.T) {
	// https://github.com/usnistgov/ACVP-Server/blob/3a7333f63/gen-val/json-files/TLS-v1.3-KDF-RFC8446/prompt.json#L428-L436
	psk := fromHex("56288B726C73829F7A3E47B103837C8139ACF552E7530C7A710B35ED41191698")
	dhe := fromHex("EFFE9EC26AA29FD750DFA6A10B944D74071595B27EE88887D5E11C84590B5CC3")
	helloClientRandom := fromHex("E9137679E582BA7C1DB41CF725F86C6D09C8C05F297BAD9A65B552EAF524FDE4")
	helloServerRandom := fromHex("23ECCFD030790748C8F8D8A656FD98D717F1B62AF3712F97211D2070B499F98A")
	finishedClientRandom := fromHex("62A62FA75563ED4FDCAA0BC16567B314871C304ACF06B0FFC3F08C1797594D43")
	finishedServerRandom := fromHex("C750EDA6696CD101B142BD79E00E6AC8C5F2C0ABC78DD64F4D991326659E9299")

	// https://github.com/usnistgov/ACVP-Server/blob/3a7333f63/gen-val/json-files/TLS-v1.3-KDF-RFC8446/expectedResults.json#L571-L581
	clientEarlyTrafficSecret := fromHex("3272189698C3594D18F58EFA3F12B638A249515099BE7A2FA9836BABE74F0111")
	earlyExporterMasterSecret := fromHex("88E078F562CDC930219F6A5E98A1CE8C6E5F3DAC5AC516459A96F2EF8F114C66")
	clientHandshakeTrafficSecret := fromHex("B32306C3CE9932C460A1FE6C0F060593974842036B96FA45049B7352E71C2AD2")
	serverHandshakeTrafficSecret := fromHex("22787F8CA269D34BC549AC8BA19F2040938A3AA370D7CC9D60F720882B88D01B")
	clientApplicationTrafficSecret := fromHex("47D7EA08397B5871154B0FE85584BCC30A87C69E84D69B56007C5B21F76493BA")
	serverApplicationTrafficSecret := fromHex("EFBDB0C873C0480DA57307083839A8984BE25B9A8545E4FCA029940FE2800565")
	exporterMasterSecret := fromHex("8A43D787EE3804EAD4A2A5B32972F9896B696295645D7222E1FD081DDD939834")
	resumptionMasterSecret := fromHex("5F4C961329C91044011ACBECB0B289282E0E3FED045CB3EA924DFFE5FE654B3D")

	// The "Random" values are undocumented, but they are meant to be written to
	// the hash in sequence to develop the transcript.
	transcript := sha256.New()

	es := tls13.NewEarlySecret(sha256.New, psk)

	transcript.Write(helloClientRandom)

	if got := es.ClientEarlyTrafficSecret(transcript); !bytes.Equal(got, clientEarlyTrafficSecret) {
		t.Errorf("clientEarlyTrafficSecret = %x, want %x", got, clientEarlyTrafficSecret)
	}
	if got := tls13.TestingOnlyExporterSecret(es.EarlyExporterMasterSecret(transcript)); !bytes.Equal(got, earlyExporterMasterSecret) {
		t.Errorf("earlyExporterMasterSecret = %x, want %x", got, earlyExporterMasterSecret)
	}

	hs := es.HandshakeSecret(dhe)

	transcript.Write(helloServerRandom)

	if got := hs.ClientHandshakeTrafficSecret(transcript); !bytes.Equal(got, clientHandshakeTrafficSecret) {
		t.Errorf("clientHandshakeTrafficSecret = %x, want %x", got, clientHandshakeTrafficSecret)
	}
	if got := hs.ServerHandshakeTrafficSecret(transcript); !bytes.Equal(got, serverHandshakeTrafficSecret) {
		t.Errorf("serverHandshakeTrafficSecret = %x, want %x", got, serverHandshakeTrafficSecret)
	}

	ms := hs.MasterSecret()

	transcript.Write(finishedServerRandom)

	if got := ms.ClientApplicationTrafficSecret(transcript); !bytes.Equal(got, clientApplicationTrafficSecret) {
		t.Errorf("clientApplicationTrafficSecret = %x, want %x", got, clientApplicationTrafficSecret)
	}
	if got := ms.ServerApplicationTrafficSecret(transcript); !bytes.Equal(got, serverApplicationTrafficSecret) {
		t.Errorf("serverApplicationTrafficSecret = %x, want %x", got, serverApplicationTrafficSecret)
	}
	if got := tls13.TestingOnlyExporterSecret(ms.ExporterMasterSecret(transcript)); !bytes.Equal(got, exporterMasterSecret) {
		t.Errorf("exporterMasterSecret = %x, want %x", got, exporterMasterSecret)
	}

	transcript.Write(finishedClientRandom)

	if got := ms.ResumptionMasterSecret(transcript); !bytes.Equal(got, resumptionMasterSecret) {
		t.Errorf("resumptionMasterSecret = %x, want %x", got, resumptionMasterSecret)
	}
}

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

func TestKyberEncapsulate(t *testing.T) {
	dk, err := mlkem.GenerateKey768()
	if err != nil {
		t.Fatal(err)
	}
	ct, ss, err := kyberEncapsulate(dk.EncapsulationKey().Bytes())
	if err != nil {
		t.Fatal(err)
	}
	dkSS, err := kyberDecapsulate(dk, ct)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(ss, dkSS) {
		t.Fatalf("got %x, want %x", ss, dkSS)
	}
}
