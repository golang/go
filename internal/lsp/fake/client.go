// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"context"

	"golang.org/x/tools/internal/lsp/protocol"
)

// Client is an adapter that converts a *Client into an LSP Client.
type Client struct {
	*Editor

	// Hooks for testing. Add additional hooks here as needed for testing.
	onLogMessage  func(context.Context, *protocol.LogMessageParams) error
	onDiagnostics func(context.Context, *protocol.PublishDiagnosticsParams) error
}

// OnLogMessage sets the hook to run when the editor receives a log message.
func (c *Client) OnLogMessage(hook func(context.Context, *protocol.LogMessageParams) error) {
	c.mu.Lock()
	c.onLogMessage = hook
	c.mu.Unlock()
}

// OnDiagnostics sets the hook to run when the editor receives diagnostics
// published from the language server.
func (c *Client) OnDiagnostics(hook func(context.Context, *protocol.PublishDiagnosticsParams) error) {
	c.mu.Lock()
	c.onDiagnostics = hook
	c.mu.Unlock()
}

func (c *Client) ShowMessage(ctx context.Context, params *protocol.ShowMessageParams) error {
	c.mu.Lock()
	c.lastMessage = params
	c.mu.Unlock()
	return nil
}

func (c *Client) ShowMessageRequest(ctx context.Context, params *protocol.ShowMessageRequestParams) (*protocol.MessageActionItem, error) {
	return nil, nil
}

func (c *Client) LogMessage(ctx context.Context, params *protocol.LogMessageParams) error {
	c.mu.Lock()
	c.logs = append(c.logs, params)
	onLogMessage := c.onLogMessage
	c.mu.Unlock()
	if onLogMessage != nil {
		return onLogMessage(ctx, params)
	}
	return nil
}

func (c *Client) Event(ctx context.Context, event *interface{}) error {
	c.mu.Lock()
	c.events = append(c.events, event)
	c.mu.Unlock()
	return nil
}

func (c *Client) PublishDiagnostics(ctx context.Context, params *protocol.PublishDiagnosticsParams) error {
	c.mu.Lock()
	c.diagnostics = params
	onPublishDiagnostics := c.onDiagnostics
	c.mu.Unlock()
	if onPublishDiagnostics != nil {
		return onPublishDiagnostics(ctx, params)
	}
	return nil
}

func (c *Client) WorkspaceFolders(context.Context) ([]protocol.WorkspaceFolder, error) {
	return []protocol.WorkspaceFolder{}, nil
}

func (c *Client) Configuration(context.Context, *protocol.ParamConfiguration) ([]interface{}, error) {
	return []interface{}{c.configuration()}, nil
}

func (c *Client) RegisterCapability(context.Context, *protocol.RegistrationParams) error {
	return nil
}

func (c *Client) UnregisterCapability(context.Context, *protocol.UnregistrationParams) error {
	return nil
}

// ApplyEdit applies edits sent from the server. Note that as of writing gopls
// doesn't use this feature, so it is untested.
func (c *Client) ApplyEdit(ctx context.Context, params *protocol.ApplyWorkspaceEditParams) (*protocol.ApplyWorkspaceEditResponse, error) {
	if len(params.Edit.Changes) != 0 {
		return &protocol.ApplyWorkspaceEditResponse{FailureReason: "Edit.Changes is unsupported"}, nil
	}
	for _, change := range params.Edit.DocumentChanges {
		path := c.ws.URIToPath(change.TextDocument.URI)
		var edits []Edit
		for _, lspEdit := range change.Edits {
			edits = append(edits, fromProtocolTextEdit(lspEdit))
		}
		c.EditBuffer(ctx, path, edits)
	}
	return &protocol.ApplyWorkspaceEditResponse{Applied: true}, nil
}
