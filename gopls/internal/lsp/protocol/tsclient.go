// Copyright 2019-2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protocol

// Code generated from version 3.17.0 of protocol/metaModel.json.
// git hash 8de18faed635819dd2bc631d2c26ce4a18f7cf4a (as of Fri Sep 16 13:04:31 2022)
// Code generated; DO NOT EDIT.

import (
	"context"
	"encoding/json"

	"golang.org/x/tools/internal/jsonrpc2"
)

type Client interface {
	LogTrace(context.Context, *LogTraceParams) error                                           // $/logTrace
	Progress(context.Context, *ProgressParams) error                                           // $/progress
	RegisterCapability(context.Context, *RegistrationParams) error                             // client/registerCapability
	UnregisterCapability(context.Context, *UnregistrationParams) error                         // client/unregisterCapability
	Event(context.Context, *interface{}) error                                                 // telemetry/event
	PublishDiagnostics(context.Context, *PublishDiagnosticsParams) error                       // textDocument/publishDiagnostics
	LogMessage(context.Context, *LogMessageParams) error                                       // window/logMessage
	ShowDocument(context.Context, *ShowDocumentParams) (*ShowDocumentResult, error)            // window/showDocument
	ShowMessage(context.Context, *ShowMessageParams) error                                     // window/showMessage
	ShowMessageRequest(context.Context, *ShowMessageRequestParams) (*MessageActionItem, error) // window/showMessageRequest
	WorkDoneProgressCreate(context.Context, *WorkDoneProgressCreateParams) error               // window/workDoneProgress/create
	ApplyEdit(context.Context, *ApplyWorkspaceEditParams) (*ApplyWorkspaceEditResult, error)   // workspace/applyEdit
	CodeLensRefresh(context.Context) error                                                     // workspace/codeLens/refresh
	Configuration(context.Context, *ParamConfiguration) ([]LSPAny, error)                      // workspace/configuration
	WorkspaceFolders(context.Context) ([]WorkspaceFolder, error)                               // workspace/workspaceFolders
}

func clientDispatch(ctx context.Context, client Client, reply jsonrpc2.Replier, r jsonrpc2.Request) (bool, error) {
	switch r.Method() {
	case "$/logTrace":
		var params LogTraceParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := client.LogTrace(ctx, &params)
		return true, reply(ctx, nil, err) // 231
	case "$/progress":
		var params ProgressParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := client.Progress(ctx, &params)
		return true, reply(ctx, nil, err) // 231
	case "client/registerCapability":
		var params RegistrationParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := client.RegisterCapability(ctx, &params)
		return true, reply(ctx, nil, err) // 155
	case "client/unregisterCapability":
		var params UnregistrationParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := client.UnregisterCapability(ctx, &params)
		return true, reply(ctx, nil, err) // 155
	case "telemetry/event":
		var params interface{}
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := client.Event(ctx, &params)
		return true, reply(ctx, nil, err) // 231
	case "textDocument/publishDiagnostics":
		var params PublishDiagnosticsParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := client.PublishDiagnostics(ctx, &params)
		return true, reply(ctx, nil, err) // 231
	case "window/logMessage":
		var params LogMessageParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := client.LogMessage(ctx, &params)
		return true, reply(ctx, nil, err) // 231
	case "window/showDocument":
		var params ShowDocumentParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := client.ShowDocument(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil) // 146
	case "window/showMessage":
		var params ShowMessageParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := client.ShowMessage(ctx, &params)
		return true, reply(ctx, nil, err) // 231
	case "window/showMessageRequest":
		var params ShowMessageRequestParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := client.ShowMessageRequest(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil) // 146
	case "window/workDoneProgress/create":
		var params WorkDoneProgressCreateParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := client.WorkDoneProgressCreate(ctx, &params)
		return true, reply(ctx, nil, err) // 155
	case "workspace/applyEdit":
		var params ApplyWorkspaceEditParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := client.ApplyEdit(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil) // 146
	case "workspace/codeLens/refresh":
		err := client.CodeLensRefresh(ctx)
		return true, reply(ctx, nil, err) // 170
	case "workspace/configuration":
		var params ParamConfiguration
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := client.Configuration(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil) // 146
	case "workspace/workspaceFolders":
		resp, err := client.WorkspaceFolders(ctx)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil) // 165
	default:
		return false, nil
	}
}

func (s *clientDispatcher) LogTrace(ctx context.Context, params *LogTraceParams) error {
	return s.sender.Notify(ctx, "$/logTrace", params)
} // 244
func (s *clientDispatcher) Progress(ctx context.Context, params *ProgressParams) error {
	return s.sender.Notify(ctx, "$/progress", params)
} // 244
func (s *clientDispatcher) RegisterCapability(ctx context.Context, params *RegistrationParams) error {
	return s.sender.Call(ctx, "client/registerCapability", params, nil)
} // 194
func (s *clientDispatcher) UnregisterCapability(ctx context.Context, params *UnregistrationParams) error {
	return s.sender.Call(ctx, "client/unregisterCapability", params, nil)
} // 194
func (s *clientDispatcher) Event(ctx context.Context, params *interface{}) error {
	return s.sender.Notify(ctx, "telemetry/event", params)
} // 244
func (s *clientDispatcher) PublishDiagnostics(ctx context.Context, params *PublishDiagnosticsParams) error {
	return s.sender.Notify(ctx, "textDocument/publishDiagnostics", params)
} // 244
func (s *clientDispatcher) LogMessage(ctx context.Context, params *LogMessageParams) error {
	return s.sender.Notify(ctx, "window/logMessage", params)
} // 244
func (s *clientDispatcher) ShowDocument(ctx context.Context, params *ShowDocumentParams) (*ShowDocumentResult, error) {
	var result *ShowDocumentResult
	if err := s.sender.Call(ctx, "window/showDocument", params, &result); err != nil {
		return nil, err
	}
	return result, nil
} // 169
func (s *clientDispatcher) ShowMessage(ctx context.Context, params *ShowMessageParams) error {
	return s.sender.Notify(ctx, "window/showMessage", params)
} // 244
func (s *clientDispatcher) ShowMessageRequest(ctx context.Context, params *ShowMessageRequestParams) (*MessageActionItem, error) {
	var result *MessageActionItem
	if err := s.sender.Call(ctx, "window/showMessageRequest", params, &result); err != nil {
		return nil, err
	}
	return result, nil
} // 169
func (s *clientDispatcher) WorkDoneProgressCreate(ctx context.Context, params *WorkDoneProgressCreateParams) error {
	return s.sender.Call(ctx, "window/workDoneProgress/create", params, nil)
} // 194
func (s *clientDispatcher) ApplyEdit(ctx context.Context, params *ApplyWorkspaceEditParams) (*ApplyWorkspaceEditResult, error) {
	var result *ApplyWorkspaceEditResult
	if err := s.sender.Call(ctx, "workspace/applyEdit", params, &result); err != nil {
		return nil, err
	}
	return result, nil
} // 169
func (s *clientDispatcher) CodeLensRefresh(ctx context.Context) error {
	return s.sender.Call(ctx, "workspace/codeLens/refresh", nil, nil)
} // 209
func (s *clientDispatcher) Configuration(ctx context.Context, params *ParamConfiguration) ([]LSPAny, error) {
	var result []LSPAny
	if err := s.sender.Call(ctx, "workspace/configuration", params, &result); err != nil {
		return nil, err
	}
	return result, nil
} // 169
func (s *clientDispatcher) WorkspaceFolders(ctx context.Context) ([]WorkspaceFolder, error) {
	var result []WorkspaceFolder
	if err := s.sender.Call(ctx, "workspace/workspaceFolders", nil, &result); err != nil {
		return nil, err
	}
	return result, nil
} // 204
