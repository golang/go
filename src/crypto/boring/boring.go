// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build boringcrypto

// Package boring exposes functions that are only available when building with
// Go+BoringCrypto. This package is available on all targets as long as the
// Go+BoringCrypto toolchain is used. Use the Enabled function to determine
// whether the BoringCrypto core is actually in use.
//
// Any time the Go+BoringCrypto toolchain is used, the "boringcrypto" build tag
// is satisfied, so that applications can tag files that use this package.
package boring

import "crypto/internal/boring"

// Enabled reports whether BoringCrypto handles supported crypto operations.
func Enabled() bool {
	return boring.Enabled
}
