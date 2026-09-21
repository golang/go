// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js

package tool

import (
	"os"
	"syscall"
)

var signalsToForward = []os.Signal{os.Interrupt, syscall.SIGQUIT, syscall.SIGTERM}
