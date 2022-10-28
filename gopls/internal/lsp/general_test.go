// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"strings"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
)

func TestVersionMessage(t *testing.T) {
	tests := []struct {
		goVersion    int
		wantContains []string // string fragments that we expect to see
		wantType     protocol.MessageType
	}{
		{-1, nil, 0},
		{12, []string{"1.12", "not supported", "upgrade to Go 1.16", "install gopls v0.7.5"}, protocol.Error},
		{13, []string{"1.13", "will be unsupported by gopls v0.11.0", "upgrade to Go 1.16", "install gopls v0.9.5"}, protocol.Warning},
		{15, []string{"1.15", "will be unsupported by gopls v0.11.0", "upgrade to Go 1.16", "install gopls v0.9.5"}, protocol.Warning},
		{16, nil, 0},
	}

	for _, test := range tests {
		gotMsg, gotType := versionMessage(test.goVersion)

		if len(test.wantContains) == 0 && gotMsg != "" {
			t.Errorf("versionMessage(%d) = %q, want \"\"", test.goVersion, gotMsg)
		}

		for _, want := range test.wantContains {
			if !strings.Contains(gotMsg, want) {
				t.Errorf("versionMessage(%d) = %q, want containing %q", test.goVersion, gotMsg, want)
			}
		}

		if gotType != test.wantType {
			t.Errorf("versionMessage(%d) = returned message type %d, want %d", test.goVersion, gotType, test.wantType)
		}
	}
}
