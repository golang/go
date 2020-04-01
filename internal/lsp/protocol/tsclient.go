package protocol

// Package protocol contains data types and code for LSP jsonrpcs
// generated automatically from vscode-languageserver-node
// commit: 151b520c995ee3d76729b5c46258ab273d989726
// last fetched Mon Mar 30 2020 21:01:17 GMT-0400 (Eastern Daylight Time)

// Code generated (see typescript/README.md) DO NOT EDIT.

import (
	"context"
	"encoding/json"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/xcontext"
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

func ClientHandler(client Client, handler jsonrpc2.Handler) jsonrpc2.Handler {
	return func(ctx context.Context, r *jsonrpc2.Request) error {
		if ctx.Err() != nil {
			ctx := xcontext.Detach(ctx)
			return r.Reply(ctx, nil, jsonrpc2.NewErrorf(RequestCancelledError, ""))
		}
		switch r.Method {
		case "window/showMessage": // notif
			var params ShowMessageParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				return sendParseError(ctx, r, err)
			}
			err := client.ShowMessage(ctx, &params)
			return r.Reply(ctx, nil, err)
		case "window/logMessage": // notif
			var params LogMessageParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				return sendParseError(ctx, r, err)
			}
			err := client.LogMessage(ctx, &params)
			return r.Reply(ctx, nil, err)
		case "telemetry/event": // notif
			var params interface{}
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				return sendParseError(ctx, r, err)
			}
			err := client.Event(ctx, &params)
			return r.Reply(ctx, nil, err)
		case "textDocument/publishDiagnostics": // notif
			var params PublishDiagnosticsParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				return sendParseError(ctx, r, err)
			}
			err := client.PublishDiagnostics(ctx, &params)
			return r.Reply(ctx, nil, err)
		case "$/progress": // notif
			var params ProgressParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				return sendParseError(ctx, r, err)
			}
			err := client.Progress(ctx, &params)
			return r.Reply(ctx, nil, err)
		case "workspace/workspaceFolders": // req
			if r.Params != nil {
				return r.Reply(ctx, nil, jsonrpc2.NewErrorf(jsonrpc2.CodeInvalidParams, "Expected no params"))
			}
			resp, err := client.WorkspaceFolders(ctx)
			return r.Reply(ctx, resp, err)
		case "workspace/configuration": // req
			var params ParamConfiguration
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				return sendParseError(ctx, r, err)
			}
			resp, err := client.Configuration(ctx, &params)
			return r.Reply(ctx, resp, err)
		case "window/workDoneProgress/create": // req
			var params WorkDoneProgressCreateParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				return sendParseError(ctx, r, err)
			}
			err := client.WorkDoneProgressCreate(ctx, &params)
			return r.Reply(ctx, nil, err)
		case "client/registerCapability": // req
			var params RegistrationParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				return sendParseError(ctx, r, err)
			}
			err := client.RegisterCapability(ctx, &params)
			return r.Reply(ctx, nil, err)
		case "client/unregisterCapability": // req
			var params UnregistrationParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				return sendParseError(ctx, r, err)
			}
			err := client.UnregisterCapability(ctx, &params)
			return r.Reply(ctx, nil, err)
		case "window/showMessageRequest": // req
			var params ShowMessageRequestParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				return sendParseError(ctx, r, err)
			}
			resp, err := client.ShowMessageRequest(ctx, &params)
			return r.Reply(ctx, resp, err)
		case "workspace/applyEdit": // req
			var params ApplyWorkspaceEditParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				return sendParseError(ctx, r, err)
			}
			resp, err := client.ApplyEdit(ctx, &params)
			return r.Reply(ctx, resp, err)
		default:
			return handler(ctx, r)

		}
	}
}

type clientDispatcher struct {
	*jsonrpc2.Conn
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
	if err := s.Conn.Call(ctx, "workspace/workspaceFolders", nil, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *clientDispatcher) Configuration(ctx context.Context, params *ParamConfiguration) ([]interface{}, error) {
	var result []interface{}
	if err := s.Conn.Call(ctx, "workspace/configuration", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *clientDispatcher) WorkDoneProgressCreate(ctx context.Context, params *WorkDoneProgressCreateParams) error {
	return s.Conn.Call(ctx, "window/workDoneProgress/create", params, nil) // Call, not Notify
}

func (s *clientDispatcher) RegisterCapability(ctx context.Context, params *RegistrationParams) error {
	return s.Conn.Call(ctx, "client/registerCapability", params, nil) // Call, not Notify
}

func (s *clientDispatcher) UnregisterCapability(ctx context.Context, params *UnregistrationParams) error {
	return s.Conn.Call(ctx, "client/unregisterCapability", params, nil) // Call, not Notify
}

func (s *clientDispatcher) ShowMessageRequest(ctx context.Context, params *ShowMessageRequestParams) (*MessageActionItem /*MessageActionItem | null*/, error) {
	var result *MessageActionItem /*MessageActionItem | null*/
	if err := s.Conn.Call(ctx, "window/showMessageRequest", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *clientDispatcher) ApplyEdit(ctx context.Context, params *ApplyWorkspaceEditParams) (*ApplyWorkspaceEditResponse, error) {
	var result *ApplyWorkspaceEditResponse
	if err := s.Conn.Call(ctx, "workspace/applyEdit", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
