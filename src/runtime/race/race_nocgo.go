// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && race && !cgo && ((linux && (amd64 || arm64 || loong64 || ppc64le || riscv64 || s390x)) || ((freebsd || netbsd) && amd64) || (windows && amd64))

package race

import _ "runtime/cgo"
