// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/tag"
)

func (s *Server) signatureHelp(ctx context.Context, params *protocol.SignatureHelpParams) (*protocol.SignatureHelp, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view, err := s.session.ViewOf(uri)
	if err != nil {
		return nil, err
	}
	snapshot := view.Snapshot()
	f, err := view.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	info, err := source.SignatureHelp(ctx, snapshot, f, params.Position)
	if err != nil {
		log.Print(ctx, "no signature help", tag.Of("At", params.Position), tag.Of("Failure", err))
		return nil, nil
	}
	return toProtocolSignatureHelp(info), nil
}

func toProtocolSignatureHelp(info *source.SignatureInformation) *protocol.SignatureHelp {
	return &protocol.SignatureHelp{
		ActiveParameter: float64(info.ActiveParameter),
		ActiveSignature: 0, // there is only ever one possible signature
		Signatures: []protocol.SignatureInformation{
			{
				Label:         info.Label,
				Documentation: info.Documentation,
				Parameters:    toProtocolParameterInformation(info.Parameters),
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
