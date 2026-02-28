// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls13

import (
	"bytes"
	"crypto/internal/fips140"
	_ "crypto/internal/fips140/check"
	"crypto/internal/fips140/sha256"
	"errors"
)

func init() {
	fips140.CAST("TLSv1.3-SHA2-256", func() error {
		input := []byte{
			0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
			0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
		}
		want := []byte{
			0x78, 0x20, 0x71, 0x75, 0x52, 0xfd, 0x47, 0x67,
			0xe1, 0x07, 0x5c, 0x83, 0x74, 0x2e, 0x49, 0x43,
			0xf7, 0xe3, 0x08, 0x6a, 0x2a, 0xcb, 0x96, 0xc7,
			0xa3, 0x1f, 0xe3, 0x23, 0x56, 0x6e, 0x14, 0x5b,
		}
		es := NewEarlySecret(sha256.New, nil)
		hs := es.HandshakeSecret(nil)
		ms := hs.MasterSecret()
		transcript := sha256.New()
		transcript.Write(input)
		if got := ms.ResumptionMasterSecret(transcript); !bytes.Equal(got, want) {
			return errors.New("unexpected result")
		}
		return nil
	})
}
