// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build race && ((linux && (amd64 || arm64 || loong64 || ppc64le || s390x)) || ((freebsd || netbsd || openbsd || windows) && amd64))

package race

// This file merely ensures that we link in runtime/cgo in race build,
// this in turn ensures that runtime uses pthread_create to create threads.
// The prebuilt race runtime lives in race_GOOS_GOARCH.syso.
// Calls to the runtime are done directly from src/runtime/race.go.

// On darwin we always use system DLLs to create threads,
// so we use race_darwin_$GOARCH.go to provide the syso-derived
// symbol information without needing to invoke cgo.
// This allows -race to be used on Mac systems without a C toolchain.

// void __race_unused_func(void);
import "C"
