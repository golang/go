// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

func (s *Server) signatureHelp(ctx context.Context, params *protocol.TextDocumentPositionParams) (*protocol.SignatureHelp, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view := s.session.ViewOf(uri)
	f, m, err := getGoFile(ctx, view, uri)
	if err != nil {
		return nil, err
	}
	spn, err := m.PointSpan(params.Position)
	if err != nil {
		return nil, err
	}
	rng, err := spn.Range(m.Converter)
	if err != nil {
		return nil, err
	}
	info, err := source.SignatureHelp(ctx, f, rng.Start)
	if err != nil {
		s.session.Logger().Infof(ctx, "no signature help for %s:%v:%v : %s", uri, int(params.Position.Line), int(params.Position.Character), err)
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
