// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (linux && !amd64.v3) || darwin || freebsd || netbsd || openbsd || windows

package race

import _ "runtime/race/internal/amd64v1"
