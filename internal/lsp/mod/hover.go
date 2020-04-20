package mod

import (
	"bytes"
	"context"
	"fmt"
	"go/token"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

func Hover(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle, position protocol.Position) (*protocol.Hover, error) {
	realURI, _ := snapshot.View().ModFiles()
	// Only get hover information on the go.mod for the view.
	if realURI == "" || fh.Identity().URI != realURI {
		return nil, nil
	}
	ctx, done := event.Start(ctx, "mod.Hover")
	defer done()

	file, m, why, err := snapshot.ModHandle(ctx, fh).Why(ctx)
	if err != nil {
		return nil, err
	}
	// Get the position of the cursor.
	spn, err := m.PointSpan(position)
	if err != nil {
		return nil, err
	}
	hoverRng, err := spn.Range(m.Converter)
	if err != nil {
		return nil, err
	}

	var req *modfile.Require
	var startPos, endPos int
	for _, r := range file.Require {
		dep := []byte(r.Mod.Path)
		s, e := r.Syntax.Start.Byte, r.Syntax.End.Byte
		i := bytes.Index(m.Content[s:e], dep)
		if i == -1 {
			continue
		}
		// Shift the start position to the location of the
		// dependency within the require statement.
		startPos, endPos = s+i, s+i+len(dep)
		if token.Pos(startPos) <= hoverRng.Start && hoverRng.Start <= token.Pos(endPos) {
			req = r
			break
		}
	}
	if req == nil || why == nil {
		return nil, nil
	}
	explanation, ok := why[req.Mod.Path]
	if !ok {
		return nil, nil
	}
	// Get the range to highlight for the hover.
	line, col, err := m.Converter.ToPosition(startPos)
	if err != nil {
		return nil, err
	}
	start := span.NewPoint(line, col, startPos)

	line, col, err = m.Converter.ToPosition(endPos)
	if err != nil {
		return nil, err
	}
	end := span.NewPoint(line, col, endPos)

	spn = span.New(fh.Identity().URI, start, end)
	rng, err := m.Range(spn)
	if err != nil {
		return nil, err
	}
	options := snapshot.View().Options()
	explanation = formatExplanation(explanation, req, options)
	return &protocol.Hover{
		Contents: protocol.MarkupContent{
			Kind:  options.PreferredContentFormat,
			Value: explanation,
		},
		Range: rng,
	}, nil
}

func formatExplanation(text string, req *modfile.Require, options source.Options) string {
	text = strings.TrimSuffix(text, "\n")
	splt := strings.Split(text, "\n")
	length := len(splt)

	var b strings.Builder
	// Write the heading as an H3.
	b.WriteString("##" + splt[0])
	if options.PreferredContentFormat == protocol.Markdown {
		b.WriteString("\n\n")
	} else {
		b.WriteRune('\n')
	}

	// If the explanation is 2 lines, then it is of the form:
	// # golang.org/x/text/encoding
	// (main module does not need package golang.org/x/text/encoding)
	if length == 2 {
		b.WriteString(splt[1])
		return b.String()
	}

	imp := splt[length-1]
	target := imp
	if strings.ToLower(options.LinkTarget) == "pkg.go.dev" {
		target = strings.Replace(target, req.Mod.Path, req.Mod.String(), 1)
	}
	target = fmt.Sprintf("https://%s/%s", options.LinkTarget, target)

	b.WriteString("This module is necessary because ")
	msg := fmt.Sprintf("[%s](%s) is imported in", imp, target)
	b.WriteString(msg)

	// If the explanation is 3 lines, then it is of the form:
	// # golang.org/x/tools
	// modtest
	// golang.org/x/tools/go/packages
	if length == 3 {
		msg := fmt.Sprintf(" `%s`.", splt[1])
		b.WriteString(msg)
		return b.String()
	}

	// If the explanation is more than 3 lines, then it is of the form:
	// # golang.org/x/text/language
	// rsc.io/quote
	// rsc.io/sampler
	// golang.org/x/text/language
	b.WriteString(":\n```text")
	dash := ""
	for _, imp := range splt[1 : length-1] {
		dash += "-"
		b.WriteString("\n" + dash + " " + imp)
	}
	b.WriteString("\n```")
	return b.String()
}
