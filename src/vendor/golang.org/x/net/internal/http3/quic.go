// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"crypto/tls"
	"slices"

	"golang.org/x/net/quic"
)

func newQUICConfig(config *quic.Config, tlsConfig *tls.Config) *quic.Config {
	config = config.Clone()
	if config == nil {
		config = &quic.Config{}
	}
	// tlsConfig may be nil: net/http.Transport.TLSClientConfig is nil
	// by default, and initEndpoint passes it to us unchanged.
	if tlsConfig == nil {
		tlsConfig = &tls.Config{NextProtos: []string{"h3"}}
	} else if !slices.Equal(tlsConfig.NextProtos, []string{"h3"}) {
		tlsConfig = tlsConfig.Clone()
		tlsConfig.NextProtos = []string{"h3"}
	}
	config.TLSConfig = tlsConfig
	return config
}
