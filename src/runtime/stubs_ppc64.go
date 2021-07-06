// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux
// +build linux

package runtime

// Called from assembly only; declared for go vet.
func load_g()
func save_g()
func reginit()

//go:noescape
func callCgoSigaction(sig uintptr, new, old *sigactiont) int32
