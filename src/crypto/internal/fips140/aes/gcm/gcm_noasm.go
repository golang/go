// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (!amd64 && !s390x && !ppc64 && !ppc64le && !arm64) || purego

package gcm

func checkGenericIsExpected() {}

type gcmPlatformData struct{}

func initGCM(g *GCM) {}

func seal(out []byte, g *GCM, nonce, plaintext, data []byte) {
	sealGeneric(out, g, nonce, plaintext, data)
}

func open(out []byte, g *GCM, nonce, ciphertext, data []byte) error {
	return openGeneric(out, g, nonce, ciphertext, data)
}
