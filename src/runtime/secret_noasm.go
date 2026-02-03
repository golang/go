// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !arm64 && !amd64

package runtime

func secretEraseRegisters() {
	throw("runtime/secret.Do not supported yet")
}
