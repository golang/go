// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ((darwin && arm64) || (darwin && amd64 && ios)) && x509omitbundledroots
// +build darwin,arm64 darwin,amd64,ios
// +build x509omitbundledroots

// This file provides the loadSystemRoots func when the
// "x509omitbundledroots" build tag has disabled bundling a copy,
// which currently on happens on darwin/arm64 (root_darwin_arm64.go).
// This then saves 256 KiB of binary size and another 560 KiB of
// runtime memory size retaining the parsed roots forever. Constrained
// environments can construct minimal x509 root CertPools on the fly
// in the crypto/tls.Config.VerifyPeerCertificate hook.

package x509

import "errors"

func loadSystemRoots() (*CertPool, error) {
	return nil, errors.New("x509: system root bundling disabled")
}

func (c *Certificate) systemVerify(opts *VerifyOptions) (chains [][]*Certificate, err error) {
	return nil, nil
}
