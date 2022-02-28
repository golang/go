// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux

package runtime

// This is needed for vet
//go:noescape
func callCgoSigaction(sig uintptr, new, old *sigactiont) int32
