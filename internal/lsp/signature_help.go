// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

func toProtocolSignatureHelp(info *source.SignatureInformation) *protocol.SignatureHelp {
	if info == nil {
		return &protocol.SignatureHelp{}
	}
	return &protocol.SignatureHelp{
		ActiveParameter: float64(info.ActiveParameter),
		ActiveSignature: 0, // there is only ever one possible signature
		Signatures: []protocol.SignatureInformation{
			{
				Label:      info.Label,
				Parameters: toProtocolParameterInformation(info.Parameters),
			},
		},
	}
}

func toProtocolParameterInformation(info []source.ParameterInformation) []protocol.ParameterInformation {
	var result []protocol.ParameterInformation
	for _, p := range info {
		result = append(result, protocol.ParameterInformation{
			Label: p.Label,
		})
	}
	return result
}
