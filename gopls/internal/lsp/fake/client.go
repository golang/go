// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"context"
	"encoding/json"
	"fmt"

	"golang.org/x/tools/gopls/internal/lsp/glob"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
)

// ClientHooks are a set of optional hooks called during handling of
// the corresponding client method (see protocol.Client for the
// LSP server-to-client RPCs) in order to make test expectations
// awaitable.
type ClientHooks struct {
	OnLogMessage             func(context.Context, *protocol.LogMessageParams) error
	OnDiagnostics            func(context.Context, *protocol.PublishDiagnosticsParams) error
	OnWorkDoneProgressCreate func(context.Context, *protocol.WorkDoneProgressCreateParams) error
	OnProgress               func(context.Context, *protocol.ProgressParams) error
	OnShowMessage            func(context.Context, *protocol.ShowMessageParams) error
	OnShowMessageRequest     func(context.Context, *protocol.ShowMessageRequestParams) error
	OnRegisterCapability     func(context.Context, *protocol.RegistrationParams) error
	OnUnregisterCapability   func(context.Context, *protocol.UnregistrationParams) error
	OnApplyEdit              func(context.Context, *protocol.ApplyWorkspaceEditParams) error
}

// Client is an adapter that converts an *Editor into an LSP Client. It mostly
// delegates functionality to hooks that can be configured by tests.
type Client struct {
	editor         *Editor
	hooks          ClientHooks
	skipApplyEdits bool // don't apply edits from ApplyEdit downcalls to Editor
}

func (c *Client) CodeLensRefresh(context.Context) error { return nil }

func (c *Client) InlayHintRefresh(context.Context) error { return nil }

func (c *Client) DiagnosticRefresh(context.Context) error { return nil }

func (c *Client) InlineValueRefresh(context.Context) error { return nil }

func (c *Client) SemanticTokensRefresh(context.Context) error { return nil }

func (c *Client) LogTrace(context.Context, *protocol.LogTraceParams) error { return nil }

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
		if item.Section == "gopls" {
			config := c.editor.Config()
			results[i] = makeSettings(c.editor.sandbox, config)
		}
	}
	return results, nil
}

func (c *Client) RegisterCapability(ctx context.Context, params *protocol.RegistrationParams) error {
	if c.hooks.OnRegisterCapability != nil {
		if err := c.hooks.OnRegisterCapability(ctx, params); err != nil {
			return err
		}
	}
	// Update file watching patterns.
	//
	// TODO(rfindley): We could verify more here, like verify that the
	// registration ID is distinct, and that the capability is not currently
	// registered.
	for _, registration := range params.Registrations {
		if registration.Method == "workspace/didChangeWatchedFiles" {
			// Marshal and unmarshal to interpret RegisterOptions as
			// DidChangeWatchedFilesRegistrationOptions.
			raw, err := json.Marshal(registration.RegisterOptions)
			if err != nil {
				return fmt.Errorf("marshaling registration options: %v", err)
			}
			var opts protocol.DidChangeWatchedFilesRegistrationOptions
			if err := json.Unmarshal(raw, &opts); err != nil {
				return fmt.Errorf("unmarshaling registration options: %v", err)
			}
			var globs []*glob.Glob
			for _, watcher := range opts.Watchers {
				// TODO(rfindley): honor the watch kind.
				g, err := glob.Parse(watcher.GlobPattern)
				if err != nil {
					return fmt.Errorf("error parsing glob pattern %q: %v", watcher.GlobPattern, err)
				}
				globs = append(globs, g)
			}
			c.editor.mu.Lock()
			c.editor.watchPatterns = globs
			c.editor.mu.Unlock()
		}
	}
	return nil
}

func (c *Client) UnregisterCapability(ctx context.Context, params *protocol.UnregistrationParams) error {
	if c.hooks.OnUnregisterCapability != nil {
		return c.hooks.OnUnregisterCapability(ctx, params)
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

func (c *Client) ApplyEdit(ctx context.Context, params *protocol.ApplyWorkspaceEditParams) (*protocol.ApplyWorkspaceEditResult, error) {
	if len(params.Edit.Changes) != 0 {
		return &protocol.ApplyWorkspaceEditResult{FailureReason: "Edit.Changes is unsupported"}, nil
	}
	if c.hooks.OnApplyEdit != nil {
		if err := c.hooks.OnApplyEdit(ctx, params); err != nil {
			return nil, err
		}
	}
	if !c.skipApplyEdits {
		for _, change := range params.Edit.DocumentChanges {
			if err := c.editor.applyDocumentChange(ctx, change); err != nil {
				return nil, err
			}
		}
	}
	return &protocol.ApplyWorkspaceEditResult{Applied: true}, nil
}
