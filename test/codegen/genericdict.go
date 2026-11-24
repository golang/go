// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

type readByte struct{ b byte }

func (r *readByte) read() byte { return r.b }

type singletonReader interface {
	*readByte
	read() byte
}

//go:noinline
func readSingleton[T singletonReader](r T) byte {
	// amd64:-"CALL"
	// arm64:-"CALL"
	return r.read() + 1
}

func singleton(r *readByte) byte { return readSingleton(r) }
