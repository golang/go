// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is required to prevent compiler errors
// when the package built with CGO_ENABLED=0.
// Otherwise the compiler says:
// pkg/net/fd_poll_runtime.go:15: missing function body
