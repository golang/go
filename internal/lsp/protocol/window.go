// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the corresponding structures to the
// "Window" messages part of the LSP specification.

package protocol

type ShowMessageParams struct {
	/**
	 * The message type. See {@link MessageType}.
	 */
	Type MessageType `json:"type"`

	/**
	 * The actual message.
	 */
	Message string `json:"message"`
}

type MessageType float64

const (
	/**
	 * An error message.
	 */
	Error MessageType = 1
	/**
	* A warning message.
	 */
	Warning MessageType = 2
	/**
	* An information message.
	 */
	Info MessageType = 3
	/**
	* A log message.
	 */
	Log MessageType = 4
)

type ShowMessageRequestParams struct {
	/**
	 * The message type. See {@link MessageType}.
	 */
	Type MessageType `json:"type"`

	/**
	* The actual message.
	 */
	Message string `json:"message"`

	/**
	 * The message action items to present.
	 */
	Actions []MessageActionItem `json:"actions,omitempty"`
}

type MessageActionItem struct {
	/**
	 * A short title like 'Retry', 'Open Log' etc.
	 */
	Title string
}

type LogMessageParams struct {
	/**
	 * The message type. See {@link MessageType}.
	 */
	Type MessageType `json:"type"`

	/**
	* The actual message.
	 */
	Message string `json:"message"`
}
