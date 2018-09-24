// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the corresponding structures to the
// "Workspace" part of the LSP specification.

package protocol

type WorkspaceFolder struct {
	/**
	 * The associated URI for this workspace folder.
	 */
	URI string `json:"uri"`

	/**
	 * The name of the workspace folder. Defaults to the
	 * uri's basename.
	 */
	Name string `json:"name"`
}

type DidChangeWorkspaceFoldersParams struct {
	/**
	 * The actual workspace folder change event.
	 */
	Event WorkspaceFoldersChangeEvent `json:"event"`
}

/**
 * The workspace folder change event.
 */
type WorkspaceFoldersChangeEvent struct {
	/**
	 * The array of added workspace folders
	 */
	Added []WorkspaceFolder `json:"added"`

	/**
	 * The array of the removed workspace folders
	 */
	Removed []WorkspaceFolder `json:"removed"`
}

type DidChangeConfigurationParams struct {
	/**
	 * The actual changed settings
	 */
	Settings interface{} `json:"settings"`
}

type ConfigurationParams struct {
	Items []ConfigurationItem `json:"items"`
}

type ConfigurationItem struct {
	/**
	 * The scope to get the configuration section for.
	 */
	ScopeURI string `json:"scopeURI,omitempty"`

	/**
	 * The configuration section asked for.
	 */
	Section string `json:"section,omitempty"`
}

type DidChangeWatchedFilesParams struct {
	/**
	 * The actual file events.
	 */
	Changes []FileEvent `json:"changes"`
}

/**
 * An event describing a file change.
 */
type FileEvent struct {
	/**
	 * The file's URI.
	 */
	URI DocumentURI `json:"uri"`
	/**
	 * The change type.
	 */
	Type float64 `json:"type"`
}

/**
 * The file event type.
 */
type FileChangeType float64

const (
	/**
	 * The file got created.
	 */
	Created FileChangeType = 1
	/**
	 * The file got changed.
	 */
	Changed FileChangeType = 2
	/**
	 * The file got deleted.
	 */
	Deleted FileChangeType = 3
)

/**
 * Describe options to be used when registering for text document change events.
 */
type DidChangeWatchedFilesRegistrationOptions struct {
	/**
	 * The watchers to register.
	 */
	Watchers []FileSystemWatcher `json:"watchers"`
}

type FileSystemWatcher struct {
	/**
	 * The  glob pattern to watch
	 */
	GlobPattern string `json:"globPattern"`

	/**
	 * The kind of events of interest. If omitted it defaults
	 * to WatchKind.Create | WatchKind.Change | WatchKind.Delete
	 * which is 7.
	 */
	Kind float64 `json:"kind,omitempty"`
}

type WatchKind float64

const (
	/**
	 * Interested in create events.
	 */
	Create WatchKind = 1

	/**
	 * Interested in change events
	 */
	Change WatchKind = 2

	/**
	 * Interested in delete events
	 */
	Delete WatchKind = 4
)

/**
 * The parameters of a Workspace Symbol Request.
 */
type WorkspaceSymbolParams struct {
	/**
	 * A non-empty query string
	 */
	Query string `json:"query"`
}

type ExecuteCommandParams struct {

	/**
	 * The identifier of the actual command handler.
	 */
	Command string `json:"command"`
	/**
	 * Arguments that the command should be invoked with.
	 */
	Arguments []interface{} `json:"arguments,omitempty"`
}

/**
 * Execute command registration options.
 */
type ExecuteCommandRegistrationOptions struct {
	/**
	 * The commands to be executed on the server
	 */
	Commands []string `json:"commands"`
}

type ApplyWorkspaceEditParams struct {
	/**
	 * An optional label of the workspace edit. This label is
	 * presented in the user interface for example on an undo
	 * stack to undo the workspace edit.
	 */
	Label string `json:"label,omitempty"`

	/**
	 * The edits to apply.
	 */
	Edit WorkspaceEdit `json:"edit"`
}

type ApplyWorkspaceEditResponse struct {
	/**
	 * Indicates whether the edit was applied or not.
	 */
	Applied bool `json:"applied"`
}
