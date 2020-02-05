package tests

import (
	"fmt"
	"strings"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
)

func DiffSignatures(spn span.Span, want, got *protocol.SignatureHelp) string {
	decorate := func(f string, args ...interface{}) string {
		return fmt.Sprintf("Invalid signature at %s: %s", spn, fmt.Sprintf(f, args...))
	}

	if len(got.Signatures) != 1 {
		return decorate("wanted 1 signature, got %d", len(got.Signatures))
	}

	if got.ActiveSignature != 0 {
		return decorate("wanted active signature of 0, got %d", int(got.ActiveSignature))
	}

	if want.ActiveParameter != got.ActiveParameter {
		return decorate("wanted active parameter of %d, got %d", want.ActiveParameter, int(got.ActiveParameter))
	}

	gotSig := got.Signatures[int(got.ActiveSignature)]

	if want.Signatures[0].Label != got.Signatures[0].Label {
		return decorate("wanted label %q, got %q", want.Signatures[0].Label, got.Signatures[0].Label)
	}

	var paramParts []string
	for _, p := range gotSig.Parameters {
		paramParts = append(paramParts, p.Label)
	}
	paramsStr := strings.Join(paramParts, ", ")
	if !strings.Contains(gotSig.Label, paramsStr) {
		return decorate("expected signature %q to contain params %q", gotSig.Label, paramsStr)
	}

	return ""
}
