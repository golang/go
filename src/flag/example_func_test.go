// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flag_test

import (
	"flag"
	"fmt"
	"net"
)

func ExampleFunc() {
	fs := flag.NewFlagSet("ExampleFunc", flag.ExitOnError)
	var ip net.IP
	fs.Func("ip", "`IP address` to parse", func(s string) error {
		return ip.UnmarshalText([]byte(s))
	})

	fs.Parse([]string{"-ip", "127.0.0.1"})
	fmt.Printf(`{ip: %v, loopback: %t}`, ip, ip.IsLoopback())

	// Output:
	// {ip: 127.0.0.1, loopback: true}
}
