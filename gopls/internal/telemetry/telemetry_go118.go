// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.19
// +build !go1.19

package telemetry

import "golang.org/x/tools/gopls/internal/lsp/protocol"

func Start() {
}

func RecordClientInfo(params *protocol.ParamInitialize) {
}

func RecordViewGoVersion(x int) {
}
