// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package exec_test

import (
	"os"
	"syscall"
)

var (
	quitSignal os.Signal = syscall.SIGQUIT
	pipeSignal os.Signal = syscall.SIGPIPE
)
