// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsprpc_test

import (
	"context"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/internal/testenv"

	. "golang.org/x/tools/gopls/internal/lsp/lsprpc"
)

type initServer struct {
	protocol.Server

	params *protocol.ParamInitialize
}

func (s *initServer) Initialize(ctx context.Context, params *protocol.ParamInitialize) (*protocol.InitializeResult, error) {
	s.params = params
	return &protocol.InitializeResult{}, nil
}

func TestGoEnvMiddleware(t *testing.T) {
	testenv.NeedsGo1Point(t, 13)

	ctx := context.Background()

	server := &initServer{}
	env := new(TestEnv)
	defer env.Shutdown(t)
	l, _ := env.serve(ctx, t, staticServerBinder(server))
	mw, err := GoEnvMiddleware()
	if err != nil {
		t.Fatal(err)
	}
	binder := mw(NewForwardBinder(l.Dialer()))
	l, _ = env.serve(ctx, t, binder)
	conn := env.dial(ctx, t, l.Dialer(), noopBinder, true)
	dispatch := protocol.ServerDispatcherV2(conn)
	initParams := &protocol.ParamInitialize{}
	initParams.InitializationOptions = map[string]interface{}{
		"env": map[string]interface{}{
			"GONOPROXY": "example.com",
		},
	}
	if _, err := dispatch.Initialize(ctx, initParams); err != nil {
		t.Fatal(err)
	}

	if server.params == nil {
		t.Fatalf("initialize params are unset")
	}
	envOpts := server.params.InitializationOptions.(map[string]interface{})["env"].(map[string]interface{})

	// Check for an arbitrary Go variable. It should be set.
	if _, ok := envOpts["GOPRIVATE"]; !ok {
		t.Errorf("Go environment variable GOPRIVATE unset in initialization options")
	}
	// Check that the variable present in our user config was not overwritten.
	if got, want := envOpts["GONOPROXY"], "example.com"; got != want {
		t.Errorf("GONOPROXY=%q, want %q", got, want)
	}
}
