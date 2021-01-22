// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux
// +build openssl
// +build !android
// +build !cmd_go_bootstrap
// +build !msan

package boring

import (
	"crypto/internal/boring/openssl"
)

type (
	GoRSA   = openssl.GoRSA
	GoECKey = openssl.GoECKey
)

func newExternalCrypto() externalCrypto {
	return openssl.NewOpenSSL()
}
