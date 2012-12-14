// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build plan9 windows

package main

import (
	"os"
)

var signalsToIgnore = []os.Signal{os.Interrupt}
