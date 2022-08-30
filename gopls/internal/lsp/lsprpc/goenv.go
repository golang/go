// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsprpc

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/gocommand"
	jsonrpc2_v2 "golang.org/x/tools/internal/jsonrpc2_v2"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
)

func GoEnvMiddleware() (Middleware, error) {
	return BindHandler(func(delegate jsonrpc2_v2.Handler) jsonrpc2_v2.Handler {
		return jsonrpc2_v2.HandlerFunc(func(ctx context.Context, req *jsonrpc2_v2.Request) (interface{}, error) {
			if req.Method == "initialize" {
				if err := addGoEnvToInitializeRequestV2(ctx, req); err != nil {
					event.Error(ctx, "adding go env to initialize", err)
				}
			}
			return delegate.Handle(ctx, req)
		})
	}), nil
}

func addGoEnvToInitializeRequestV2(ctx context.Context, req *jsonrpc2_v2.Request) error {
	var params protocol.ParamInitialize
	if err := json.Unmarshal(req.Params, &params); err != nil {
		return err
	}
	var opts map[string]interface{}
	switch v := params.InitializationOptions.(type) {
	case nil:
		opts = make(map[string]interface{})
	case map[string]interface{}:
		opts = v
	default:
		return fmt.Errorf("unexpected type for InitializationOptions: %T", v)
	}
	envOpt, ok := opts["env"]
	if !ok {
		envOpt = make(map[string]interface{})
	}
	env, ok := envOpt.(map[string]interface{})
	if !ok {
		return fmt.Errorf("env option is %T, expected a map", envOpt)
	}
	goenv, err := getGoEnv(ctx, env)
	if err != nil {
		return err
	}
	// We don't want to propagate GOWORK unless explicitly set since that could mess with
	// path inference during cmd/go invocations, see golang/go#51825.
	_, goworkSet := os.LookupEnv("GOWORK")
	for govar, value := range goenv {
		if govar == "GOWORK" && !goworkSet {
			continue
		}
		env[govar] = value
	}
	opts["env"] = env
	params.InitializationOptions = opts
	raw, err := json.Marshal(params)
	if err != nil {
		return fmt.Errorf("marshaling updated options: %v", err)
	}
	req.Params = json.RawMessage(raw)
	return nil
}

func getGoEnv(ctx context.Context, env map[string]interface{}) (map[string]string, error) {
	var runEnv []string
	for k, v := range env {
		runEnv = append(runEnv, fmt.Sprintf("%s=%s", k, v))
	}
	runner := gocommand.Runner{}
	output, err := runner.Run(ctx, gocommand.Invocation{
		Verb: "env",
		Args: []string{"-json"},
		Env:  runEnv,
	})
	if err != nil {
		return nil, err
	}
	envmap := make(map[string]string)
	if err := json.Unmarshal(output.Bytes(), &envmap); err != nil {
		return nil, err
	}
	return envmap, nil
}
