// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package holds the race detector .syso for
// amd64 architectures with GOAMD64<v3.

//go:build amd64 && ((linux && !amd64.v3) || darwin || freebsd || netbsd || openbsd || windows)

package amd64v1
