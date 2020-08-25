// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package boring exposes functions that are only available when building with
// Go+BoringCrypto. This package is available on all targets as long as the
// Go+BoringCrypto toolchain is used. Use the Enabled function to determine
// whether the BoringCrypto core is actually in use.
package boring

import "crypto/internal/boring"

// Enabled reports whether BoringCrypto handles supported crypto operations.
func Enabled() bool {
	return boring.Enabled
}
