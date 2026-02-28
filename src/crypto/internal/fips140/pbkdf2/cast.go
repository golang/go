// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbkdf2

import (
	"bytes"
	"crypto/internal/fips140"
	_ "crypto/internal/fips140/check"
	"crypto/internal/fips140/sha256"
	"errors"
)

func init() {
	// Per IG 10.3.A:
	//   "if the module implements an approved PBKDF (SP 800-132), the module
	//    shall perform a CAST, at minimum, on the derivation of the Master
	//   Key (MK) as specified in Section 5.3 of SP 800-132"
	//   "The Iteration Count parameter does not need to be among those
	//   supported by the module in the approved mode but shall be at least
	//   two."
	fips140.CAST("PBKDF2", func() error {
		salt := []byte{
			0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11,
			0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19,
		}
		want := []byte{
			0xC7, 0x58, 0x76, 0xC0, 0x71, 0x1C, 0x29, 0x75,
			0x2D, 0x3A, 0xA6, 0xDF, 0x29, 0x96,
		}

		mk, err := Key(sha256.New, "password", salt, 2, 14)
		if err != nil {
			return err
		}
		if !bytes.Equal(mk, want) {
			return errors.New("unexpected result")
		}

		return nil
	})
}
