// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows plan9 nacl

package runtime_test

import "os"

// sigquit is the signal to send to kill a hanging testdata program.
// On Unix we send SIGQUIT, but on non-Unix we only have os.Kill.
var sigquit = os.Kill
