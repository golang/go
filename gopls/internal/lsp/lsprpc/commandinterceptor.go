// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsprpc

import (
	"context"
	"encoding/json"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	jsonrpc2_v2 "golang.org/x/tools/internal/jsonrpc2_v2"
)

// HandlerMiddleware is a middleware that only modifies the jsonrpc2 handler.
type HandlerMiddleware func(jsonrpc2_v2.Handler) jsonrpc2_v2.Handler

// BindHandler transforms a HandlerMiddleware into a Middleware.
func BindHandler(hmw HandlerMiddleware) Middleware {
	return Middleware(func(binder jsonrpc2_v2.Binder) jsonrpc2_v2.Binder {
		return BinderFunc(func(ctx context.Context, conn *jsonrpc2_v2.Connection) jsonrpc2_v2.ConnectionOptions {
			opts := binder.Bind(ctx, conn)
			opts.Handler = hmw(opts.Handler)
			return opts
		})
	})
}

func CommandInterceptor(command string, run func(*protocol.ExecuteCommandParams) (interface{}, error)) Middleware {
	return BindHandler(func(delegate jsonrpc2_v2.Handler) jsonrpc2_v2.Handler {
		return jsonrpc2_v2.HandlerFunc(func(ctx context.Context, req *jsonrpc2_v2.Request) (interface{}, error) {
			if req.Method == "workspace/executeCommand" {
				var params protocol.ExecuteCommandParams
				if err := json.Unmarshal(req.Params, &params); err == nil {
					if params.Command == command {
						return run(&params)
					}
				}
			}

			return delegate.Handle(ctx, req)
		})
	})
}
