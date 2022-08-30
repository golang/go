// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsprpc_test

import (
	"context"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/protocol"

	. "golang.org/x/tools/gopls/internal/lsp/lsprpc"
)

func TestCommandInterceptor(t *testing.T) {
	const command = "foo"
	caught := false
	intercept := func(_ *protocol.ExecuteCommandParams) (interface{}, error) {
		caught = true
		return map[string]interface{}{}, nil
	}

	ctx := context.Background()
	env := new(TestEnv)
	defer env.Shutdown(t)
	mw := CommandInterceptor(command, intercept)
	l, _ := env.serve(ctx, t, mw(noopBinder))
	conn := env.dial(ctx, t, l.Dialer(), noopBinder, false)

	params := &protocol.ExecuteCommandParams{
		Command: command,
	}
	var res interface{}
	err := conn.Call(ctx, "workspace/executeCommand", params).Await(ctx, &res)
	if err != nil {
		t.Fatal(err)
	}
	if !caught {
		t.Errorf("workspace/executeCommand was not intercepted")
	}
}
