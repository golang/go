// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protocol

// Package protocol contains data types and code for LSP jsonrpcs
// generated automatically from vscode-languageserver-node
// commit: dae62de921d25964e8732411ca09e532dde992f5
// last fetched Sat Jan 23 2021 16:14:55 GMT-0500 (Eastern Standard Time)

// Code generated (see typescript/README.md) DO NOT EDIT.

import (
	"context"
	"encoding/json"

	"golang.org/x/tools/internal/jsonrpc2"
	errors "golang.org/x/xerrors"
)

type Client interface {
	ShowMessage(context.Context, *ShowMessageParams) error
	LogMessage(context.Context, *LogMessageParams) error
	Event(context.Context, *interface{}) error
	PublishDiagnostics(context.Context, *PublishDiagnosticsParams) error
	Progress(context.Context, *ProgressParams) error
	WorkspaceFolders(context.Context) ([]WorkspaceFolder /*WorkspaceFolder[] | null*/, error)
	Configuration(context.Context, *ParamConfiguration) ([]interface{}, error)
	WorkDoneProgressCreate(context.Context, *WorkDoneProgressCreateParams) error
	RegisterCapability(context.Context, *RegistrationParams) error
	UnregisterCapability(context.Context, *UnregistrationParams) error
	ShowMessageRequest(context.Context, *ShowMessageRequestParams) (*MessageActionItem /*MessageActionItem | null*/, error)
	ApplyEdit(context.Context, *ApplyWorkspaceEditParams) (*ApplyWorkspaceEditResponse, error)
}

func clientDispatch(ctx context.Context, client Client, reply jsonrpc2.Replier, r jsonrpc2.Request) (bool, error) {
	switch r.Method() {
	case "window/showMessage": // notif
		var params ShowMessageParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := client.ShowMessage(ctx, &params)
		return true, reply(ctx, nil, err)
	case "window/logMessage": // notif
		var params LogMessageParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := client.LogMessage(ctx, &params)
		return true, reply(ctx, nil, err)
	case "telemetry/event": // notif
		var params interface{}
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := client.Event(ctx, &params)
		return true, reply(ctx, nil, err)
	case "textDocument/publishDiagnostics": // notif
		var params PublishDiagnosticsParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := client.PublishDiagnostics(ctx, &params)
		return true, reply(ctx, nil, err)
	case "$/progress": // notif
		var params ProgressParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := client.Progress(ctx, &params)
		return true, reply(ctx, nil, err)
	case "workspace/workspaceFolders": // req
		if len(r.Params()) > 0 {
			return true, reply(ctx, nil, errors.Errorf("%w: expected no params", jsonrpc2.ErrInvalidParams))
		}
		resp, err := client.WorkspaceFolders(ctx)
		return true, reply(ctx, resp, err)
	case "workspace/configuration": // req
		var params ParamConfiguration
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := client.Configuration(ctx, &params)
		return true, reply(ctx, resp, err)
	case "window/workDoneProgress/create": // req
		var params WorkDoneProgressCreateParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := client.WorkDoneProgressCreate(ctx, &params)
		return true, reply(ctx, nil, err)
	case "client/registerCapability": // req
		var params RegistrationParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := client.RegisterCapability(ctx, &params)
		return true, reply(ctx, nil, err)
	case "client/unregisterCapability": // req
		var params UnregistrationParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := client.UnregisterCapability(ctx, &params)
		return true, reply(ctx, nil, err)
	case "window/showMessageRequest": // req
		var params ShowMessageRequestParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := client.ShowMessageRequest(ctx, &params)
		return true, reply(ctx, resp, err)
	case "workspace/applyEdit": // req
		var params ApplyWorkspaceEditParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := client.ApplyEdit(ctx, &params)
		return true, reply(ctx, resp, err)

	default:
		return false, nil
	}
}

func (s *clientDispatcher) ShowMessage(ctx context.Context, params *ShowMessageParams) error {
	return s.Conn.Notify(ctx, "window/showMessage", params)
}

func (s *clientDispatcher) LogMessage(ctx context.Context, params *LogMessageParams) error {
	return s.Conn.Notify(ctx, "window/logMessage", params)
}

func (s *clientDispatcher) Event(ctx context.Context, params *interface{}) error {
	return s.Conn.Notify(ctx, "telemetry/event", params)
}

func (s *clientDispatcher) PublishDiagnostics(ctx context.Context, params *PublishDiagnosticsParams) error {
	return s.Conn.Notify(ctx, "textDocument/publishDiagnostics", params)
}

func (s *clientDispatcher) Progress(ctx context.Context, params *ProgressParams) error {
	return s.Conn.Notify(ctx, "$/progress", params)
}
func (s *clientDispatcher) WorkspaceFolders(ctx context.Context) ([]WorkspaceFolder /*WorkspaceFolder[] | null*/, error) {
	var result []WorkspaceFolder /*WorkspaceFolder[] | null*/
	if err := Call(ctx, s.Conn, "workspace/workspaceFolders", nil, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *clientDispatcher) Configuration(ctx context.Context, params *ParamConfiguration) ([]interface{}, error) {
	var result []interface{}
	if err := Call(ctx, s.Conn, "workspace/configuration", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *clientDispatcher) WorkDoneProgressCreate(ctx context.Context, params *WorkDoneProgressCreateParams) error {
	return Call(ctx, s.Conn, "window/workDoneProgress/create", params, nil) // Call, not Notify
}

func (s *clientDispatcher) RegisterCapability(ctx context.Context, params *RegistrationParams) error {
	return Call(ctx, s.Conn, "client/registerCapability", params, nil) // Call, not Notify
}

func (s *clientDispatcher) UnregisterCapability(ctx context.Context, params *UnregistrationParams) error {
	return Call(ctx, s.Conn, "client/unregisterCapability", params, nil) // Call, not Notify
}

func (s *clientDispatcher) ShowMessageRequest(ctx context.Context, params *ShowMessageRequestParams) (*MessageActionItem /*MessageActionItem | null*/, error) {
	var result *MessageActionItem /*MessageActionItem | null*/
	if err := Call(ctx, s.Conn, "window/showMessageRequest", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *clientDispatcher) ApplyEdit(ctx context.Context, params *ApplyWorkspaceEditParams) (*ApplyWorkspaceEditResponse, error) {
	var result *ApplyWorkspaceEditResponse
	if err := Call(ctx, s.Conn, "workspace/applyEdit", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
