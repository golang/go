// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the corresponding structures to the
// "Client" part of the LSP specification.

package protocol

/**
 * General parameters to register for a capability.
 */
type Registration struct {
	/**
	 * The id used to register the request. The id can be used to deregister
	 * the request again.
	 */
	ID string `json:"id"`

	/**
	 * The method / capability to register for.
	 */
	Method string `json:"method"`

	/**
	 * Options necessary for the registration.
	 */
	RegisterOptions interface{} `json:"registerOptions,omitempty"`
}

type RegistrationParams struct {
	Registrations []Registration `json:"registrations"`
}

type TextDocumentRegistrationOptions struct {
	/**
	 * A document selector to identify the scope of the registration. If set to null
	 * the document selector provided on the client side will be used.
	 */
	DocumentSelector *DocumentSelector `json:"documentSelector"`
}

/**
 * General parameters to unregister a capability.
 */
type Unregistration struct {
	/**
	 * The id used to unregister the request or notification. Usually an id
	 * provided during the register request.
	 */
	ID string `json:"id"`

	/**
	 * The method / capability to unregister for.
	 */
	Method string `json:"method"`
}

type UnregistrationParams struct {
	Unregisterations []Unregistration `json:"unregisterations"`
}
