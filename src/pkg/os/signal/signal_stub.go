// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build plan9

package signal

import "os"

const numSig = 0

func signum(sig os.Signal) int { return -1 }

func disableSignal(int) {}

func enableSignal(int) {}
