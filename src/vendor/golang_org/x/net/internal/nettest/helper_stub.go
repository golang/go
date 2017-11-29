// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build nacl plan9

package nettest

import (
	"fmt"
	"runtime"
)

func maxOpenFiles() int {
	return defaultMaxOpenFiles
}

func supportsRawIPSocket() (string, bool) {
	return fmt.Sprintf("not supported on %s", runtime.GOOS), false
}

func supportsIPv6MulticastDeliveryOnLoopback() bool {
	return false
}

func causesIPv6Crash() bool {
	return false
}

func protocolNotSupported(err error) bool {
	return false
}
