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
		fromBuild    bool
		wantContains []string // string fragments that we expect to see
		wantType     protocol.MessageType
	}{
		{-1, false, nil, 0},
		{12, false, []string{"1.12", "not supported", "upgrade to Go 1.18", "install gopls v0.7.5"}, protocol.Error},
		{13, false, []string{"1.13", "not supported", "upgrade to Go 1.18", "install gopls v0.9.5"}, protocol.Error},
		{15, false, []string{"1.15", "not supported", "upgrade to Go 1.18", "install gopls v0.9.5"}, protocol.Error},
		{15, true, []string{"Gopls was built with Go version 1.15", "not supported", "upgrade to Go 1.18", "install gopls v0.9.5"}, protocol.Error},
		{16, false, []string{"1.16", "will be unsupported by gopls v0.13.0", "upgrade to Go 1.18", "install gopls v0.11.0"}, protocol.Warning},
		{17, false, []string{"1.17", "will be unsupported by gopls v0.13.0", "upgrade to Go 1.18", "install gopls v0.11.0"}, protocol.Warning},
		{17, true, []string{"Gopls was built with Go version 1.17", "will be unsupported by gopls v0.13.0", "upgrade to Go 1.18", "install gopls v0.11.0"}, protocol.Warning},
	}

	for _, test := range tests {
		gotMsg, gotType := versionMessage(test.goVersion, test.fromBuild)

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
