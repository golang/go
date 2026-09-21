// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sigchanyzer

import (
	"os"
	"os/signal"
)

func _() {
	c := make(chan os.Signal)
	signal.Notify(c, os.Interrupt) // ERROR "misuse of unbuffered os.Signal channel as argument to signal.Notify"
}
