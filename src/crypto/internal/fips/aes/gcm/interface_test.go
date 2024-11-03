// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gcm_test

import (
	"crypto/cipher"
	"crypto/internal/fips/aes/gcm"
)

var _ cipher.AEAD = (*gcm.GCM)(nil)
