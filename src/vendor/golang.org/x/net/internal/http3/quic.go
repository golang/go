// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"crypto/tls"

	"golang.org/x/net/quic"
)

func initConfig(config *quic.Config) *quic.Config {
	if config == nil {
		config = &quic.Config{}
	}

	// maybeCloneTLSConfig clones the user-provided tls.Config (but only once)
	// prior to us modifying it.
	needCloneTLSConfig := true
	maybeCloneTLSConfig := func() *tls.Config {
		if needCloneTLSConfig {
			config.TLSConfig = config.TLSConfig.Clone()
			needCloneTLSConfig = false
		}
		return config.TLSConfig
	}

	if config.TLSConfig == nil {
		config.TLSConfig = &tls.Config{}
		needCloneTLSConfig = false
	}
	if config.TLSConfig.MinVersion == 0 {
		maybeCloneTLSConfig().MinVersion = tls.VersionTLS13
	}
	if config.TLSConfig.NextProtos == nil {
		maybeCloneTLSConfig().NextProtos = []string{"h3"}
	}
	return config
}
