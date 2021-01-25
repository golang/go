// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux,amd64
// +build cgo
// +build !openssl
// +build !android
// +build !cmd_go_bootstrap
// +build !msan

package boring

import (
	"crypto/internal/boring/boringcrypto"
)

type (
	GoRSA   = boringcrypto.GoRSA
	GoECKey = boringcrypto.GoECKey
)

func newExternalCrypto() externalCrypto {
	return boringcrypto.NewBoringCrypto()
}
