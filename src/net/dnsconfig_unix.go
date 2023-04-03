// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !js && !windows

// Read system DNS config from /etc/resolv.conf

package net

func dnsReadConfig(filename string) *dnsConfig {
	return parseResolvConf(filename)
}
