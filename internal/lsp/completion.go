package lsp

import (
	"sort"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

func toProtocolCompletionItems(items []source.CompletionItem) []protocol.CompletionItem {
	var results []protocol.CompletionItem
	sort.Slice(items, func(i, j int) bool {
		return items[i].Score > items[j].Score
	})
	for _, item := range items {
		results = append(results, protocol.CompletionItem{
			Label:  item.Label,
			Detail: item.Detail,
			Kind:   float64(toProtocolCompletionItemKind(item.Kind)),
		})
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
