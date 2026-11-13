// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package upki

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/binary"
	"errors"
	"math/big"
	"slices"
	"testing"
	"time"

	"golang.org/x/crypto/cryptobyte"
)

func TestExtractSCTs(t *testing.T) {
	t.Parallel()

	sctA := SCT{logID: filledLogID(0x11), timestamp: 1700000000000}
	sctB := SCT{logID: filledLogID(0x22), timestamp: 1700000123456}
	tests := []struct {
		name string
		cert *x509.Certificate
		want []SCT
	}{
		{
			name: "absent",
			cert: &x509.Certificate{},
		},
		{
			name: "multiple",
			cert: certificateWithSCTExtension(t, serializedSCT(sctA), serializedSCT(sctB)),
			want: []SCT{sctA, sctB},
		},
		{
			// An SCT whose version byte is not 0 (v1) is skipped without
			// parsing its body, so everything after the version is arbitrary.
			name: "unknown version ignored",
			cert: certificateWithSCTExtension(t,
				serializedSCT(sctA),
				append([]byte{1}, []byte("arbitrary")...),
				serializedSCT(sctB)),
			want: []SCT{sctA, sctB},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			got, err := ExtractSCTs(test.cert)
			if err != nil {
				t.Fatalf("ExtractSCTs: %v", err)
			}
			if !slices.Equal(got, test.want) {
				t.Fatalf("ExtractSCTs = %v, want %v", got, test.want)
			}
		})
	}
}

func TestExtractSCTsMalformed(t *testing.T) {
	t.Parallel()

	valid := sctExtensionValue(t, serializedSCT(SCT{}))
	// The extension value nests four length-delimited layers: an ASN.1
	// OCTET STRING, a uint16-prefixed SCT list, a uint16-prefixed
	// serialized SCT, and inside it the fixed fields (version, 32-byte log
	// ID, 8-byte timestamp) followed by uint16-prefixed CT extensions. Each
	// case truncates or pads one layer.
	tests := []struct {
		name  string
		value []byte
	}{
		{
			// OCTET STRING tag 0x04 claiming 5 content bytes, with one.
			name:  "outer wrapper truncated",
			value: []byte{0x04, 0x05, 0x00},
		},
		{
			// A byte after the end of the OCTET STRING.
			name:  "outer wrapper trailing data",
			value: append(bytes.Clone(valid), 0),
		},
		{
			// SCT list length 5 (0x0005), with one byte following.
			name:  "SCT list truncated",
			value: asn1OctetString(t, []byte{0, 5, 0}),
		},
		{
			// SCT list length 3 holding a serialized SCT length of 5
			// (0x0005) with one byte following.
			name:  "serialized SCT length truncated",
			value: asn1OctetString(t, []byte{0, 3, 0, 5, 0}),
		},
		{
			// A zero-length serialized SCT, so no version byte.
			name:  "version missing",
			value: sctExtensionValue(t, nil),
		},
		{
			// Version 0 (v1) followed by one byte of the 32-byte log ID.
			name:  "log ID truncated",
			value: sctExtensionValue(t, []byte{0, 1}),
		},
		{
			// Version and log ID, with no timestamp.
			name:  "timestamp truncated",
			value: sctExtensionValue(t, append([]byte{0}, make([]byte, 32)...)),
		},
		{
			// Version, log ID and timestamp, with no CT extensions length.
			name:  "extensions truncated",
			value: sctExtensionValue(t, append(append([]byte{0}, make([]byte, 32)...), make([]byte, 8)...)),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			cert := &x509.Certificate{Extensions: []pkix.Extension{{
				Id:    embeddedSCTOID,
				Value: test.value,
			}}}
			if _, err := ExtractSCTs(cert); !errors.Is(err, errMalformedSCTExtension) {
				t.Fatalf("ExtractSCTs error = %v, want errMalformedSCTExtension", err)
			}
		})
	}
}

func TestRawSerial(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		serial *big.Int
		want   []byte
	}{
		{"ordinary", big.NewInt(42), []byte{42}},
		{
			// A positive serial whose top bit is set. DER encodes it with
			// a leading 0x00 so the INTEGER is not read as negative, and
			// rawSerial must keep that byte, unlike big.Int.Bytes.
			name:   "leading sign byte",
			serial: new(big.Int).SetBytes([]byte{0x80, 0xab, 0xcd, 0xef}),
			want:   []byte{0, 0x80, 0xab, 0xcd, 0xef},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			got, err := rawSerial(newTestCertificate(t, test.serial))
			if err != nil {
				t.Fatalf("rawSerial: %v", err)
			}
			if !bytes.Equal(got, test.want) {
				t.Fatalf("rawSerial = %x, want %x", got, test.want)
			}
		})
	}
}

func TestRawSerialMalformed(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		tbs  []byte
	}{
		{
			// Not a SEQUENCE (tag 0x30), so TBSCertificate cannot be entered.
			name: "outer sequence",
			tbs:  []byte{0xde, 0xad},
		},
		{
			// An empty SEQUENCE: no version and no serialNumber INTEGER.
			name: "serial",
			tbs:  []byte{0x30, 0x00},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			cert := &x509.Certificate{RawTBSCertificate: test.tbs}
			if _, err := rawSerial(cert); !errors.Is(err, errMalformedTBSCertificate) {
				t.Fatalf("rawSerial error = %v, want errMalformedTBSCertificate", err)
			}
		})
	}
}

func certificateWithSCTExtension(t *testing.T, scts ...[]byte) *x509.Certificate {
	t.Helper()

	return &x509.Certificate{Extensions: []pkix.Extension{{
		Id:    embeddedSCTOID,
		Value: sctExtensionValue(t, scts...),
	}}}
}

func sctExtensionValue(t *testing.T, scts ...[]byte) []byte {
	t.Helper()

	var list []byte
	for _, sct := range scts {
		list = binary.BigEndian.AppendUint16(list, uint16(len(sct)))
		list = append(list, sct...)
	}
	inner := binary.BigEndian.AppendUint16(nil, uint16(len(list)))
	inner = append(inner, list...)

	return asn1OctetString(t, inner)
}

func serializedSCT(sct SCT) []byte {
	serialized := []byte{0} // version v1
	serialized = append(serialized, sct.logID[:]...)
	serialized = binary.BigEndian.AppendUint64(serialized, uint64(sct.timestamp))
	// ExtractSCTs reads through the CT extensions and then discards the
	// remainder of the length-delimited SCT, so no signature is needed.
	serialized = append(serialized, 0, 0) // empty CtExtensions

	return serialized
}

func asn1OctetString(t *testing.T, contents []byte) []byte {
	t.Helper()

	var builder cryptobyte.Builder
	builder.AddASN1OctetString(contents)
	encoded, err := builder.Bytes()
	if err != nil {
		t.Fatal(err)
	}

	return encoded
}

func filledLogID(value byte) logID {
	var id logID
	for i := range id {
		id[i] = value
	}

	return id
}

func newTestCertificate(t *testing.T, serial *big.Int) *x509.Certificate {
	t.Helper()

	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	template := &x509.Certificate{
		SerialNumber: serial,
		Subject:      pkix.Name{CommonName: "test"},
		NotBefore:    time.Unix(0, 0),
		NotAfter:     time.Unix(1<<31, 0),
	}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	cert, err := x509.ParseCertificate(der)
	if err != nil {
		t.Fatal(err)
	}

	return cert
}
