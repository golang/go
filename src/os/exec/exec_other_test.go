// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !unix && !windows

package exec_test

import "os"

var (
	quitSignal os.Signal = nil
	pipeSignal os.Signal = nil
)
