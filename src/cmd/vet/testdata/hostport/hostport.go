// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the hostport checker.

package hostport

import (
	"fmt"
	"net"
)

func _(host string, port int) {
	addr := fmt.Sprintf("%s:%d", host, port) // ERROR "address format .%s:%d. does not work with IPv6"
	net.Dial("tcp", addr)
}
