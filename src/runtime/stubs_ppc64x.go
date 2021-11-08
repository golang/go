// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64le || ppc64

package runtime

// Called from assembly only; declared for go vet.
func load_g()
func save_g()
func reginit()

// Spills/loads arguments in registers to/from an internal/abi.RegArgs
// respectively. Does not follow the Go ABI.
func spillArgs()
func unspillArgs()
