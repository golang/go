package diff

import (
	"bytes"
	"sort"

	"golang.org/x/tools/internal/span"
)

func init() {
	ApplyEdits = applyEdits
}

func applyEdits(before string, edits []TextEdit) string {
	// Preconditions:
	//   - all of the edits apply to before
	//   - and all the spans for each TextEdit have the same URI

	// copy edits so we don't make a mess of the caller's slice
	s := make([]TextEdit, len(edits))
	copy(s, edits)
	edits = s

	// TODO(matloob): Initialize the Converter Once?
	var conv span.Converter = span.NewContentConverter("", []byte(before))
	offset := func(point span.Point) int {
		if point.HasOffset() {
			return point.Offset()
		}
		offset, err := conv.ToOffset(point.Line(), point.Column())
		if err != nil {
			panic(err)
		}
		return offset
	}

	// sort the copy
	sort.Slice(edits, func(i, j int) bool { return offset(edits[i].Span.Start()) < offset(edits[j].Span.Start()) })

	var after bytes.Buffer
	beforeOffset := 0
	for _, edit := range edits {
		if offset(edit.Span.Start()) < beforeOffset {
			panic("overlapping edits") // TODO(matloob): ApplyEdits doesn't return an error. What do we do?
		} else if offset(edit.Span.Start()) > beforeOffset {
			after.WriteString(before[beforeOffset:offset(edit.Span.Start())])
			beforeOffset = offset(edit.Span.Start())
		}
		// offset(edit.Span.Start) is now equal to beforeOffset
		after.WriteString(edit.NewText)
		beforeOffset += offset(edit.Span.End()) - offset(edit.Span.Start())
	}
	if beforeOffset < len(before) {
		after.WriteString(before[beforeOffset:])
		beforeOffset = len(before[beforeOffset:]) // just to preserve invariants
	}
	return after.String()
}
