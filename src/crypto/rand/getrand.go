// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (js && wasm) || wasip1 || windows

package rand

import "crypto/rand/internal/getrand"

var randReader = rngReader{}

type rngReader struct{}

func (r rngReader) Read(b []byte) (int, error) {
	if err := getrand.GetRandom(b); err != nil {
		return 0, err
	}
	return len(b), nil
}
