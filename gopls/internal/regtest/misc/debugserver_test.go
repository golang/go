// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"net/http"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/command"
	"golang.org/x/tools/gopls/internal/lsp/protocol"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

func TestStartDebugging(t *testing.T) {
	WithOptions(
		Modes(Forwarded),
	).Run(t, "", func(t *testing.T, env *Env) {
		args, err := command.MarshalArgs(command.DebuggingArgs{})
		if err != nil {
			t.Fatal(err)
		}
		params := &protocol.ExecuteCommandParams{
			Command:   command.StartDebugging.ID(),
			Arguments: args,
		}
		var result command.DebuggingResult
		env.ExecuteCommand(params, &result)
		if got, want := len(result.URLs), 2; got != want {
			t.Fatalf("got %d urls, want %d; urls: %#v", got, want, result.URLs)
		}
		for i, u := range result.URLs {
			resp, err := http.Get(u)
			if err != nil {
				t.Errorf("getting url #%d (%q): %v", i, u, err)
				continue
			}
			defer resp.Body.Close()
			if got, want := resp.StatusCode, http.StatusOK; got != want {
				t.Errorf("debug server #%d returned HTTP %d, want %d", i, got, want)
			}
		}
	})
}
