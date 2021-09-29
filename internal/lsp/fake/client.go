// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"context"
	"fmt"

	"golang.org/x/tools/internal/lsp/protocol"
)

// ClientHooks are called to handle the corresponding client LSP method.
type ClientHooks struct {
	OnLogMessage             func(context.Context, *protocol.LogMessageParams) error
	OnDiagnostics            func(context.Context, *protocol.PublishDiagnosticsParams) error
	OnWorkDoneProgressCreate func(context.Context, *protocol.WorkDoneProgressCreateParams) error
	OnProgress               func(context.Context, *protocol.ProgressParams) error
	OnShowMessage            func(context.Context, *protocol.ShowMessageParams) error
	OnShowMessageRequest     func(context.Context, *protocol.ShowMessageRequestParams) error
	OnRegistration           func(context.Context, *protocol.RegistrationParams) error
	OnUnregistration         func(context.Context, *protocol.UnregistrationParams) error
}

// Client is an adapter that converts an *Editor into an LSP Client. It mosly
// delegates functionality to hooks that can be configured by tests.
type Client struct {
	editor *Editor
	hooks  ClientHooks
}

func (c *Client) ShowMessage(ctx context.Context, params *protocol.ShowMessageParams) error {
	if c.hooks.OnShowMessage != nil {
		return c.hooks.OnShowMessage(ctx, params)
	}
	return nil
}

func (c *Client) ShowMessageRequest(ctx context.Context, params *protocol.ShowMessageRequestParams) (*protocol.MessageActionItem, error) {
	if c.hooks.OnShowMessageRequest != nil {
		if err := c.hooks.OnShowMessageRequest(ctx, params); err != nil {
			return nil, err
		}
	}
	if len(params.Actions) == 0 || len(params.Actions) > 1 {
		return nil, fmt.Errorf("fake editor cannot handle multiple action items")
	}
	return &params.Actions[0], nil
}

func (c *Client) LogMessage(ctx context.Context, params *protocol.LogMessageParams) error {
	if c.hooks.OnLogMessage != nil {
		return c.hooks.OnLogMessage(ctx, params)
	}
	return nil
}

func (c *Client) Event(ctx context.Context, event *interface{}) error {
	return nil
}

func (c *Client) PublishDiagnostics(ctx context.Context, params *protocol.PublishDiagnosticsParams) error {
	if c.hooks.OnDiagnostics != nil {
		return c.hooks.OnDiagnostics(ctx, params)
	}
	return nil
}

func (c *Client) WorkspaceFolders(context.Context) ([]protocol.WorkspaceFolder, error) {
	return []protocol.WorkspaceFolder{}, nil
}

func (c *Client) Configuration(_ context.Context, p *protocol.ParamConfiguration) ([]interface{}, error) {
	results := make([]interface{}, len(p.Items))
	for i, item := range p.Items {
		if item.Section != "gopls" {
			continue
		}
		results[i] = c.editor.configuration()
	}
	return results, nil
}

func (c *Client) RegisterCapability(ctx context.Context, params *protocol.RegistrationParams) error {
	if c.hooks.OnRegistration != nil {
		return c.hooks.OnRegistration(ctx, params)
	}
	return nil
}

func (c *Client) UnregisterCapability(ctx context.Context, params *protocol.UnregistrationParams) error {
	if c.hooks.OnUnregistration != nil {
		return c.hooks.OnUnregistration(ctx, params)
	}
	return nil
}

func (c *Client) Progress(ctx context.Context, params *protocol.ProgressParams) error {
	if c.hooks.OnProgress != nil {
		return c.hooks.OnProgress(ctx, params)
	}
	return nil
}

func (c *Client) WorkDoneProgressCreate(ctx context.Context, params *protocol.WorkDoneProgressCreateParams) error {
	if c.hooks.OnWorkDoneProgressCreate != nil {
		return c.hooks.OnWorkDoneProgressCreate(ctx, params)
	}
	return nil
}

func (c *Client) ShowDocument(context.Context, *protocol.ShowDocumentParams) (*protocol.ShowDocumentResult, error) {
	return nil, nil
}

// ApplyEdit applies edits sent from the server.
func (c *Client) ApplyEdit(ctx context.Context, params *protocol.ApplyWorkspaceEditParams) (*protocol.ApplyWorkspaceEditResult, error) {
	if len(params.Edit.Changes) != 0 {
		return &protocol.ApplyWorkspaceEditResult{FailureReason: "Edit.Changes is unsupported"}, nil
	}
	for _, change := range params.Edit.DocumentChanges {
		if err := c.editor.applyProtocolEdit(ctx, change); err != nil {
			return nil, err
		}
	}
	return &protocol.ApplyWorkspaceEditResult{Applied: true}, nil
}
