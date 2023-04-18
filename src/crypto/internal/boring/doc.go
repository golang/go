// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package boring provides access to BoringCrypto implementation functions.
// Check the constant Enabled to find out whether BoringCrypto is available.
// If BoringCrypto is not available, the functions in this package all panic.
package boring

// Enabled reports whether BoringCrypto is available.
// When enabled is false, all functions in this package panic.
//
// BoringCrypto is only available on linux/amd64 systems.
const Enabled = available

// A BigInt is the raw words from a BigInt.
// This definition allows us to avoid importing math/big.
// Conversion between BigInt and *big.Int is in crypto/internal/boring/bbig.
type BigInt []uint
