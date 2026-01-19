// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Keep in sync with notboring.go and crypto/internal/boring/boring.go.
//go:build boringcrypto && linux && (amd64 || arm64) && !android && !msan && cgo

package fips140

const boringEnabled = true
