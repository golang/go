// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specgen

import (
	"fmt"
	"strings"
)

// ParagraphKind classifies a doc comment paragraph.
type ParagraphKind int

const (
	// SpecOwned paragraphs describe the behavioral semantics of an operation.
	SpecOwned ParagraphKind = iota

	// ImplementationNote paragraphs describe target-specific implementation
	// details (such as assembly instructions, required CPU features, emulation
	// notes, etc.).
	//
	// All implementation note paragraphs start with one of a fixed set of
	// prefixes (see [HasNotePrefix]).
	ImplementationNote

	// DirectiveComment paragraphs represent compiler or tool directives (e.g.
	// //go:noescape).
	DirectiveComment
)

func (k ParagraphKind) String() string {
	switch k {
	case SpecOwned:
		return "spec-owned paragraph"
	case ImplementationNote:
		return "implementation note"
	case DirectiveComment:
		return "directive comment"
	default:
		return fmt.Sprintf("ParagraphKind(%d)", k)
	}
}

// Paragraph is a doc comment paragraph with its classification.
type Paragraph struct {
	Kind ParagraphKind
	Text string // No leading "//" or trailing \n
}

// HasNotePrefix reports whether text starts with an implementation note prefix.
// The text should NOT include any leading "//".
func HasNotePrefix(text string) bool {
	if strings.HasPrefix(text, "Asm:") ||
		strings.HasPrefix(text, "CPU Feature:") ||
		strings.HasPrefix(text, "Deprecated:") ||
		strings.HasPrefix(text, "Performance:") ||
		strings.HasPrefix(text, "Note:") {
		return true
	}
	// "Emulated" notes are a little bit of a mess. They don't usually have any
	// content, so there's no colon. (go.dev/issue/81568)
	if strings.HasPrefix(text, "Emulated,") || text == "Emulated" || strings.HasPrefix(text, "Emulated\n") || strings.HasPrefix(text, "Emulated:") {
		return true
	}
	return false
}

// parseSpecDoc splits expanded spec doc text (without comment markers) into
// paragraphs.
//
// It reports errors to ctx if any paragraph in text is not "spec-owned". The
// spec docs are merged with implementation notes and directive comments in
// generated source. In order for this to be idempotent, we must only inject
// spec text into source code that will still be recognized as spec text when we
// need to strip it back out.
func parseSpecDoc(ctx context, text string) []Paragraph {
	text = strings.Trim(text, "\n")

	var out []Paragraph
	for para := range strings.SplitSeq(text, "\n\n") {
		// Spec docs structurally cannot contain directive comments.
		if HasNotePrefix(para) {
			ctx.errorf("spec doc contains implementation note: %s", strings.TrimSpace(para))
		} else {
			out = append(out, Paragraph{SpecOwned, para})
		}
	}

	return out
}

// FormatComment formats a slice of paragraphs into standard Go comment text.
// The result has "//"-prefixed lines and ends with "\n" unless the whole result
// is empty.
//
// FormatComment checks that [SpecOwned] paragraphs must appear strictly before
// [ImplementationNote] paragraphs, which must appear strictly before
// [DirectiveComment]s. If the ordering rule is violated, it returns an error.
func FormatComment(paras []Paragraph) (string, error) {
	if len(paras) == 0 {
		return "", nil
	}

	lastKind := SpecOwned
	for _, p := range paras {
		if p.Kind < lastKind {
			return "", fmt.Errorf("doc comment ordering violation: %s appeared after %s", p.Kind, lastKind)
		}
		lastKind = p.Kind
	}

	var buf strings.Builder
	inDirectives := false
	for i, p := range paras {
		if i > 0 && !inDirectives {
			buf.WriteString("//\n")
		}
		if p.Kind == DirectiveComment {
			buf.WriteString("//")
			buf.WriteString(p.Text)
			buf.WriteByte('\n')
			// Directives don't have paragraph separators
			inDirectives = true
		} else {
			for line := range strings.SplitSeq(p.Text, "\n") {
				if strings.HasPrefix(line, "\t") {
					buf.WriteString("//")
				} else {
					buf.WriteString("// ")
				}
				buf.WriteString(line)
				buf.WriteByte('\n')
			}
		}
	}

	return buf.String(), nil
}
