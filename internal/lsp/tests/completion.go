package tests

import (
	"bytes"
	"fmt"
	"sort"
	"strings"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

func ToProtocolCompletionItems(items []source.CompletionItem) []protocol.CompletionItem {
	var result []protocol.CompletionItem
	for _, item := range items {
		result = append(result, ToProtocolCompletionItem(item))
	}
	return result
}

func ToProtocolCompletionItem(item source.CompletionItem) protocol.CompletionItem {
	pItem := protocol.CompletionItem{
		Label:         item.Label,
		Kind:          item.Kind,
		Detail:        item.Detail,
		Documentation: item.Documentation,
		InsertText:    item.InsertText,
		TextEdit: &protocol.TextEdit{
			NewText: item.Snippet(),
		},
	}
	if pItem.InsertText == "" {
		pItem.InsertText = pItem.Label
	}
	return pItem
}

func FilterBuiltins(items []protocol.CompletionItem) []protocol.CompletionItem {
	var got []protocol.CompletionItem
	for _, item := range items {
		if isBuiltin(item.Label, item.Detail, item.Kind) {
			continue
		}
		got = append(got, item)
	}
	return got
}

func isBuiltin(label, detail string, kind protocol.CompletionItemKind) bool {
	if detail == "" && kind == protocol.ClassCompletion {
		return true
	}
	// Remaining builtin constants, variables, interfaces, and functions.
	trimmed := label
	if i := strings.Index(trimmed, "("); i >= 0 {
		trimmed = trimmed[:i]
	}
	switch trimmed {
	case "append", "cap", "close", "complex", "copy", "delete",
		"error", "false", "imag", "iota", "len", "make", "new",
		"nil", "panic", "print", "println", "real", "recover", "true":
		return true
	}
	return false
}

func CheckCompletionOrder(want, got []protocol.CompletionItem) string {
	var (
		matchedIdxs []int
		lastGotIdx  int
		inOrder     = true
	)
	for _, w := range want {
		var found bool
		for i, g := range got {
			if w.Label == g.Label && w.Detail == g.Detail && w.Kind == g.Kind {
				matchedIdxs = append(matchedIdxs, i)
				found = true
				if i < lastGotIdx {
					inOrder = false
				}
				lastGotIdx = i
				break
			}
		}
		if !found {
			return summarizeCompletionItems(-1, []protocol.CompletionItem{w}, got, "didn't find expected completion")
		}
	}

	sort.Ints(matchedIdxs)
	matched := make([]protocol.CompletionItem, 0, len(matchedIdxs))
	for _, idx := range matchedIdxs {
		matched = append(matched, got[idx])
	}

	if !inOrder {
		return summarizeCompletionItems(-1, want, matched, "completions out of order")
	}

	return ""
}

func DiffSnippets(want string, got *protocol.CompletionItem) string {
	if want == "" {
		if got != nil {
			return fmt.Sprintf("expected no snippet but got %s", got.TextEdit.NewText)
		}
	} else {
		if got == nil {
			return fmt.Sprintf("couldn't find completion matching %q", want)
		}
		if want != got.TextEdit.NewText {
			return fmt.Sprintf("expected snippet %q, got %q", want, got.TextEdit.NewText)
		}
	}
	return ""
}

func FindItem(list []protocol.CompletionItem, want source.CompletionItem) *protocol.CompletionItem {
	for _, item := range list {
		if item.Label == want.Label {
			return &item
		}
	}
	return nil
}

// DiffCompletionItems prints the diff between expected and actual completion
// test results.
func DiffCompletionItems(want, got []protocol.CompletionItem) string {
	if len(got) != len(want) {
		return summarizeCompletionItems(-1, want, got, "different lengths got %v want %v", len(got), len(want))
	}
	for i, w := range want {
		g := got[i]
		if w.Label != g.Label {
			return summarizeCompletionItems(i, want, got, "incorrect Label got %v want %v", g.Label, w.Label)
		}
		if w.Detail != g.Detail {
			return summarizeCompletionItems(i, want, got, "incorrect Detail got %v want %v", g.Detail, w.Detail)
		}
		if w.Documentation != "" && !strings.HasPrefix(w.Documentation, "@") {
			if w.Documentation != g.Documentation {
				return summarizeCompletionItems(i, want, got, "incorrect Documentation got %v want %v", g.Documentation, w.Documentation)
			}
		}
		if w.Kind != g.Kind {
			return summarizeCompletionItems(i, want, got, "incorrect Kind got %v want %v", g.Kind, w.Kind)
		}
	}
	return ""
}

func summarizeCompletionItems(i int, want, got []protocol.CompletionItem, reason string, args ...interface{}) string {
	msg := &bytes.Buffer{}
	fmt.Fprint(msg, "completion failed")
	if i >= 0 {
		fmt.Fprintf(msg, " at %d", i)
	}
	fmt.Fprint(msg, " because of ")
	fmt.Fprintf(msg, reason, args...)
	fmt.Fprint(msg, ":\nexpected:\n")
	for _, d := range want {
		fmt.Fprintf(msg, "  %v\n", d)
	}
	fmt.Fprintf(msg, "got:\n")
	for _, d := range got {
		fmt.Fprintf(msg, "  %v\n", d)
	}
	return msg.String()
}
