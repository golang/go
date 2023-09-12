// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.19
// +build !go1.19

package telemetry

import "golang.org/x/tools/gopls/internal/lsp/protocol"

func Mode() string {
	return "local"
}

func SetMode(mode string) error {
	return nil
}

func Start() {
}

func RecordClientInfo(params *protocol.ParamInitialize) {
}

func RecordViewGoVersion(x int) {
}

func AddForwardedCounters(names []string, values []int64) {
}
