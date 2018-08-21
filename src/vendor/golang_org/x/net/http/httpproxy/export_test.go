// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package httpproxy

func ExportUseProxy(cfg *Config, host string) bool {
	cfg1 := &config{
		Config: *cfg,
	}
	cfg1.init()
	return cfg1.useProxy(host)
}
