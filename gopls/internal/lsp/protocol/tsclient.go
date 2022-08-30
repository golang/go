// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Code generated (see typescript/README.md) DO NOT EDIT.

package protocol

// Package protocol contains data types and code for LSP json rpcs
// generated automatically from vscode-languageserver-node
// commit: 696f9285bf849b73745682fdb1c1feac73eb8772
// last fetched Fri Apr 01 2022 10:53:41 GMT-0400 (Eastern Daylight Time)

import (
	"context"
	"encoding/json"
	"fmt"

	"golang.org/x/tools/internal/jsonrpc2"
)

type Client interface {
	ShowMessage(context.Context, *ShowMessageParams) error
	LogMessage(context.Context, *LogMessageParams) error
	Event(context.Context, *interface{}) error
	PublishDiagnostics(context.Context, *PublishDiagnosticsParams) error
	Progress(context.Context, *ProgressParams) error
	WorkspaceFolders(context.Context) ([]WorkspaceFolder /*WorkspaceFolder[] | null*/, error)
	Configuration(context.Context, *ParamConfiguration) ([]LSPAny, error)
	WorkDoneProgressCreate(context.Context, *WorkDoneProgressCreateParams) error
	ShowDocument(context.Context, *ShowDocumentParams) (*ShowDocumentResult, error)
	RegisterCapability(context.Context, *RegistrationParams) error
	UnregisterCapability(context.Context, *UnregistrationParams) error
	ShowMessageRequest(context.Context, *ShowMessageRequestParams) (*MessageActionItem /*MessageActionItem | null*/, error)
	ApplyEdit(context.Context, *ApplyWorkspaceEditParams) (*ApplyWorkspaceEditResult, error)
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
			return true, reply(ctx, nil, fmt.Errorf("%w: expected no params", jsonrpc2.ErrInvalidParams))
		}
		resp, err := client.WorkspaceFolders(ctx)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "workspace/configuration": // req
		var params ParamConfiguration
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := client.Configuration(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "window/workDoneProgress/create": // req
		var params WorkDoneProgressCreateParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := client.WorkDoneProgressCreate(ctx, &params)
		return true, reply(ctx, nil, err)
	case "window/showDocument": // req
		var params ShowDocumentParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := client.ShowDocument(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
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
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "workspace/applyEdit": // req
		var params ApplyWorkspaceEditParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := client.ApplyEdit(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)

	default:
		return false, nil
	}
}

func (s *clientDispatcher) ShowMessage(ctx context.Context, params *ShowMessageParams) error {
	return s.sender.Notify(ctx, "window/showMessage", params)
}

func (s *clientDispatcher) LogMessage(ctx context.Context, params *LogMessageParams) error {
	return s.sender.Notify(ctx, "window/logMessage", params)
}

func (s *clientDispatcher) Event(ctx context.Context, params *interface{}) error {
	return s.sender.Notify(ctx, "telemetry/event", params)
}

func (s *clientDispatcher) PublishDiagnostics(ctx context.Context, params *PublishDiagnosticsParams) error {
	return s.sender.Notify(ctx, "textDocument/publishDiagnostics", params)
}

func (s *clientDispatcher) Progress(ctx context.Context, params *ProgressParams) error {
	return s.sender.Notify(ctx, "$/progress", params)
}
func (s *clientDispatcher) WorkspaceFolders(ctx context.Context) ([]WorkspaceFolder /*WorkspaceFolder[] | null*/, error) {
	var result []WorkspaceFolder /*WorkspaceFolder[] | null*/
	if err := s.sender.Call(ctx, "workspace/workspaceFolders", nil, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *clientDispatcher) Configuration(ctx context.Context, params *ParamConfiguration) ([]LSPAny, error) {
	var result []LSPAny
	if err := s.sender.Call(ctx, "workspace/configuration", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *clientDispatcher) WorkDoneProgressCreate(ctx context.Context, params *WorkDoneProgressCreateParams) error {
	return s.sender.Call(ctx, "window/workDoneProgress/create", params, nil) // Call, not Notify
}

func (s *clientDispatcher) ShowDocument(ctx context.Context, params *ShowDocumentParams) (*ShowDocumentResult, error) {
	var result *ShowDocumentResult
	if err := s.sender.Call(ctx, "window/showDocument", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *clientDispatcher) RegisterCapability(ctx context.Context, params *RegistrationParams) error {
	return s.sender.Call(ctx, "client/registerCapability", params, nil) // Call, not Notify
}

func (s *clientDispatcher) UnregisterCapability(ctx context.Context, params *UnregistrationParams) error {
	return s.sender.Call(ctx, "client/unregisterCapability", params, nil) // Call, not Notify
}

func (s *clientDispatcher) ShowMessageRequest(ctx context.Context, params *ShowMessageRequestParams) (*MessageActionItem /*MessageActionItem | null*/, error) {
	var result *MessageActionItem /*MessageActionItem | null*/
	if err := s.sender.Call(ctx, "window/showMessageRequest", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *clientDispatcher) ApplyEdit(ctx context.Context, params *ApplyWorkspaceEditParams) (*ApplyWorkspaceEditResult, error) {
	var result *ApplyWorkspaceEditResult
	if err := s.sender.Call(ctx, "workspace/applyEdit", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
