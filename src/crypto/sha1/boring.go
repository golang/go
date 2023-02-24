// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Extra indirection here so that when building go_bootstrap
// cmd/internal/boring is not even imported, so that we don't
// have to maintain changes to cmd/dist's deps graph.

//go:build !cmd_go_bootstrap && cgo
// +build !cmd_go_bootstrap,cgo

package sha1

import (
	"crypto/internal/boring"
	"hash"
)

const boringEnabled = boring.Enabled

func boringNewSHA1() hash.Hash { return boring.NewSHA1() }

func boringUnreachable() { boring.Unreachable() }

func boringSHA1(p []byte) [20]byte { return boring.SHA1(p) }
