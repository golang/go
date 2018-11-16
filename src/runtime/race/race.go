// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build race,linux,amd64 race,freebsd,amd64 race,netbsd,amd64 race,darwin,amd64 race,windows,amd64 race,linux,ppc64le race,linux,arm64

package race

// This file merely ensures that we link in runtime/cgo in race build,
// this is turn ensures that runtime uses pthread_create to create threads.
// The prebuilt race runtime lives in race_GOOS_GOARCH.syso.
// Calls to the runtime are done directly from src/runtime/race.go.

// void __race_unused_func(void);
import "C"
