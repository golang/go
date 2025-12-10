// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimesecret

// Package secret contains helper functions for zeroing out memory
// that is otherwise invisible to a user program in the service of
// forward secrecy. See https://en.wikipedia.org/wiki/Forward_secrecy for
// more information.
//
// This package (runtime/secret) is experimental,
// and not subject to the Go 1 compatibility promise.
// It only exists when building with the GOEXPERIMENT=runtimesecret environment variable set.
package secret
