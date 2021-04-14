// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cfg holds configuration shared by the Go command and internal/testenv.
// Definitions that don't need to be exposed outside of cmd/go should be in
// cmd/go/internal/cfg instead of this package.
package cfg

// KnownEnv is a list of environment variables that affect the operation
// of the Go command.
const KnownEnv = `
	AR
	CC
	CGO_CFLAGS
	CGO_CFLAGS_ALLOW
	CGO_CFLAGS_DISALLOW
	CGO_CPPFLAGS
	CGO_CPPFLAGS_ALLOW
	CGO_CPPFLAGS_DISALLOW
	CGO_CXXFLAGS
	CGO_CXXFLAGS_ALLOW
	CGO_CXXFLAGS_DISALLOW
	CGO_ENABLED
	CGO_FFLAGS
	CGO_FFLAGS_ALLOW
	CGO_FFLAGS_DISALLOW
	CGO_LDFLAGS
	CGO_LDFLAGS_ALLOW
	CGO_LDFLAGS_DISALLOW
	CXX
	FC
	GCCGO
	GO111MODULE
	GO386
	GOARCH
	GOARM
	GOBIN
	GOCACHE
	GOENV
	GOEXE
	GOFLAGS
	GOGCCFLAGS
	GOHOSTARCH
	GOHOSTOS
	GOINSECURE
	GOMIPS
	GOMIPS64
	GOMODCACHE
	GONOPROXY
	GONOSUMDB
	GOOS
	GOPATH
	GOPPC64
	GOPRIVATE
	GOPROXY
	GOROOT
	GOSUMDB
	GOTMPDIR
	GOTOOLDIR
	GOWASM
	GO_EXTLINK_ENABLED
	PKG_CONFIG
`
