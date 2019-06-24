package protocol

// Code generated (see typescript/README.md) DO NOT EDIT.

import (
	"context"
	"encoding/json"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp/xlog"
)

type Client interface {
	ShowMessage(context.Context, *ShowMessageParams) error
	LogMessage(context.Context, *LogMessageParams) error
	Event(context.Context, *interface{}) error
	PublishDiagnostics(context.Context, *PublishDiagnosticsParams) error
	WorkspaceFolders(context.Context) ([]WorkspaceFolder, error)
	Configuration(context.Context, *ConfigurationParams) ([]interface{}, error)
	RegisterCapability(context.Context, *RegistrationParams) error
	UnregisterCapability(context.Context, *UnregistrationParams) error
	ShowMessageRequest(context.Context, *ShowMessageRequestParams) (*MessageActionItem, error)
	ApplyEdit(context.Context, *ApplyWorkspaceEditParams) (*ApplyWorkspaceEditResponse, error)
}

func clientHandler(log xlog.Logger, client Client) jsonrpc2.Handler {
	return func(ctx context.Context, r *jsonrpc2.Request) {
		switch r.Method {
		case "$/cancelRequest":
			var params CancelParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, r, err)
				return
			}
			r.Conn().Cancel(params.ID)
		case "window/showMessage": // notif
			var params ShowMessageParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, r, err)
				return
			}
			if err := client.ShowMessage(ctx, &params); err != nil {
				log.Errorf(ctx, "%v", err)
			}
		case "window/logMessage": // notif
			var params LogMessageParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, r, err)
				return
			}
			if err := client.LogMessage(ctx, &params); err != nil {
				log.Errorf(ctx, "%v", err)
			}
		case "telemetry/event": // notif
			var params interface{}
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, r, err)
				return
			}
			if err := client.Event(ctx, &params); err != nil {
				log.Errorf(ctx, "%v", err)
			}
		case "textDocument/publishDiagnostics": // notif
			var params PublishDiagnosticsParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, r, err)
				return
			}
			if err := client.PublishDiagnostics(ctx, &params); err != nil {
				log.Errorf(ctx, "%v", err)
			}
		case "workspace/workspaceFolders": // req
			if r.Params != nil {
				r.Reply(ctx, nil, jsonrpc2.NewErrorf(jsonrpc2.CodeInvalidParams, "Expected no params"))
				return
			}
			resp, err := client.WorkspaceFolders(ctx)
			if err := r.Reply(ctx, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}
		case "workspace/configuration": // req
			var params ConfigurationParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, r, err)
				return
			}
			resp, err := client.Configuration(ctx, &params)
			if err := r.Reply(ctx, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}
		case "client/registerCapability": // req
			var params RegistrationParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, r, err)
				return
			}
			err := client.RegisterCapability(ctx, &params)
			if err := r.Reply(ctx, nil, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}
		case "client/unregisterCapability": // req
			var params UnregistrationParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, r, err)
				return
			}
			err := client.UnregisterCapability(ctx, &params)
			if err := r.Reply(ctx, nil, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}
		case "window/showMessageRequest": // req
			var params ShowMessageRequestParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, r, err)
				return
			}
			resp, err := client.ShowMessageRequest(ctx, &params)
			if err := r.Reply(ctx, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}
		case "workspace/applyEdit": // req
			var params ApplyWorkspaceEditParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, r, err)
				return
			}
			resp, err := client.ApplyEdit(ctx, &params)
			if err := r.Reply(ctx, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		default:
			if r.IsNotify() {
				r.Reply(ctx, nil, jsonrpc2.NewErrorf(jsonrpc2.CodeMethodNotFound, "method %q not found", r.Method))
			}
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
func (s *clientDispatcher) WorkspaceFolders(ctx context.Context) ([]WorkspaceFolder, error) {
	var result []WorkspaceFolder
	if err := s.Conn.Call(ctx, "workspace/workspaceFolders", nil, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *clientDispatcher) Configuration(ctx context.Context, params *ConfigurationParams) ([]interface{}, error) {
	var result []interface{}
	if err := s.Conn.Call(ctx, "workspace/configuration", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *clientDispatcher) RegisterCapability(ctx context.Context, params *RegistrationParams) error {
	return s.Conn.Call(ctx, "client/registerCapability", params, nil) // Call, not Notify
}

func (s *clientDispatcher) UnregisterCapability(ctx context.Context, params *UnregistrationParams) error {
	return s.Conn.Call(ctx, "client/unregisterCapability", params, nil) // Call, not Notify
}

func (s *clientDispatcher) ShowMessageRequest(ctx context.Context, params *ShowMessageRequestParams) (*MessageActionItem, error) {
	var result MessageActionItem
	if err := s.Conn.Call(ctx, "window/showMessageRequest", params, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (s *clientDispatcher) ApplyEdit(ctx context.Context, params *ApplyWorkspaceEditParams) (*ApplyWorkspaceEditResponse, error) {
	var result ApplyWorkspaceEditResponse
	if err := s.Conn.Call(ctx, "workspace/applyEdit", params, &result); err != nil {
		return nil, err
	}
	return &result, nil
}
