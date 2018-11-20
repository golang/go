// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"fmt"
	"sort"
	"strings"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

func toProtocolCompletionItems(items []source.CompletionItem, snippetsSupported, signatureHelpEnabled bool) []protocol.CompletionItem {
	var results []protocol.CompletionItem
	sort.Slice(items, func(i, j int) bool {
		return items[i].Score > items[j].Score
	})
	insertTextFormat := protocol.PlainTextFormat
	if snippetsSupported {
		insertTextFormat = protocol.SnippetTextFormat
	}
	for _, item := range items {
		insertText, triggerSignatureHelp := labelToProtocolSnippets(item.Label, item.Kind, insertTextFormat, signatureHelpEnabled)
		i := protocol.CompletionItem{
			Label:            item.Label,
			InsertText:       insertText,
			Detail:           item.Detail,
			Kind:             float64(toProtocolCompletionItemKind(item.Kind)),
			InsertTextFormat: insertTextFormat,
		}
		// If we are completing a function, we should trigger signature help if possible.
		if triggerSignatureHelp && signatureHelpEnabled {
			i.Command = &protocol.Command{
				Command: "editor.action.triggerParameterHints",
			}
		}
		results = append(results, i)
	}
	return results
}

func toProtocolCompletionItemKind(kind source.CompletionItemKind) protocol.CompletionItemKind {
	switch kind {
	case source.InterfaceCompletionItem:
		return protocol.InterfaceCompletion
	case source.StructCompletionItem:
		return protocol.StructCompletion
	case source.TypeCompletionItem:
		return protocol.TypeParameterCompletion // ??
	case source.ConstantCompletionItem:
		return protocol.ConstantCompletion
	case source.FieldCompletionItem:
		return protocol.FieldCompletion
	case source.ParameterCompletionItem, source.VariableCompletionItem:
		return protocol.VariableCompletion
	case source.FunctionCompletionItem:
		return protocol.FunctionCompletion
	case source.MethodCompletionItem:
		return protocol.MethodCompletion
	case source.PackageCompletionItem:
		return protocol.ModuleCompletion // ??
	default:
		return protocol.TextCompletion
	}
}

func labelToProtocolSnippets(label string, kind source.CompletionItemKind, insertTextFormat protocol.InsertTextFormat, signatureHelpEnabled bool) (string, bool) {
	switch kind {
	case source.ConstantCompletionItem:
		// The label for constants is of the format "<identifier> = <value>".
		// We should now insert the " = <value>" part of the label.
		return label[:strings.Index(label, " =")], false
	case source.FunctionCompletionItem, source.MethodCompletionItem:
		trimmed := label[:strings.Index(label, "(")]
		params := strings.Trim(label[strings.Index(label, "("):], "()")
		if params == "" {
			return label, true
		}
		// Don't add parameters or parens for the plaintext insert format.
		if insertTextFormat == protocol.PlainTextFormat {
			return trimmed, true
		}
		// If we do have signature help enabled, the user can see parameters as
		// they type in the function, so we just return empty parentheses.
		if signatureHelpEnabled {
			return trimmed + "($1)", true
		}
		// If signature help is not enabled, we should give the user parameters
		// that they can tab through. The insert text format follows the
		// specification defined by Microsoft for LSP. The "$", "}, and "\"
		// characters should be escaped.
		r := strings.NewReplacer(
			`\`, `\\`,
			`}`, `\}`,
			`$`, `\$`,
		)
		trimmed += "("
		for i, p := range strings.Split(params, ",") {
			if i != 0 {
				trimmed += ", "
			}
			trimmed += fmt.Sprintf("${%v:%v}", i+1, r.Replace(strings.Trim(p, " ")))
		}
		return trimmed + ")", false

	}
	return label, false
}
