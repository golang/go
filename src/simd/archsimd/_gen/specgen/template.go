// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specgen

import (
	"bytes"
	"cmp"
	"fmt"
	"reflect"
	"regexp"
	"simd/archsimd/_gen/specgen/specexpr"
	"slices"
	"strings"
	"text/template"
	"unicode/utf8"
)

// specTemplate wraps a parsed text/template.Template, or a plain string if no
// template actions were present.
type specTemplate struct {
	raw  string
	tmpl *template.Template
}

var templateFuncs = template.FuncMap{
	"title": func(v any) string {
		s := fmt.Sprint(v)
		if len(s) == 0 {
			return ""
		}
		return strings.ToTitle(s[:1]) + s[1:]
	},
	"lt": func(a, b any) (bool, error) {
		res, ok := compare(a, b)
		if !ok {
			return false, fmt.Errorf("incomparable types: %T and %T", a, b)
		}
		return res < 0, nil
	},
	"le": func(a, b any) (bool, error) {
		res, ok := compare(a, b)
		if !ok {
			return false, fmt.Errorf("incomparable types: %T and %T", a, b)
		}
		return res <= 0, nil
	},
	"gt": func(a, b any) (bool, error) {
		res, ok := compare(a, b)
		if !ok {
			return false, fmt.Errorf("incomparable types: %T and %T", a, b)
		}
		return res > 0, nil
	},
	"ge": func(a, b any) (bool, error) {
		res, ok := compare(a, b)
		if !ok {
			return false, fmt.Errorf("incomparable types: %T and %T", a, b)
		}
		return res >= 0, nil
	},
	"eq": func(a, b any) bool {
		if res, ok := compare(a, b); ok {
			return res == 0
		}
		return reflect.DeepEqual(a, b)
	},
	"ne": func(a, b any) bool {
		if res, ok := compare(a, b); ok {
			return res != 0
		}
		return !reflect.DeepEqual(a, b)
	},

	// lanes returns shape.Lanes() if shape is non-scalable; or else "name.Len()"
	"lanes": func(shape, name any) (string, error) {
		s, ok := shape.(specexpr.Vector)
		if !ok {
			return "", fmt.Errorf("want Vector, got %T", shape)
		}
		if !s.Scalable() {
			return s.Elems().String(), nil
		}
		return name.(string) + ".Len()", nil
	},
	"reshapeDiagram": func(x, z any) (string, error) {
		xx, ok := x.(specexpr.Vector)
		if !ok {
			return "", fmt.Errorf("want Vector, got %T", x)
		}
		zz, ok := z.(specexpr.Vector)
		if !ok {
			return "", fmt.Errorf("want Vector, got %T", z)
		}
		if xx.Width != zz.Width {
			return "", fmt.Errorf("Vectors must have equal widths, got %s and %s", xx, zz)
		}
		return reshapeDiagram(xx, zz), nil
	},
}

func toSpecNum(v any) (specexpr.Num, bool) {
	switch v := v.(type) {
	case specexpr.Num:
		return v, true
	case int:
		return specexpr.Int(v), true
	case int64:
		return specexpr.Int(v), true
	default:
		return nil, false
	}
}

func compare(a, b any) (int, bool) {
	if na, okA := toSpecNum(a); okA {
		if nb, okB := toSpecNum(b); okB {
			return na.Compare(nb)
		}
	}
	if sa, okA := a.(string); okA {
		if sb, okB := b.(string); okB {
			return cmp.Compare(sa, sb), true
		}
	}
	return 0, false
}

// reshapeDiagram returns a diagram showing how vector elements are
// reinterpreted by ReshapeToUints. For example, for Uint32x8 -> Uint64x4, it
// returns:
//
//	    x[7]      x[6]    ⋯     x[1]      x[0]
//	| 31 .. 0 | 31 .. 0 | ⋯ | 31 .. 0 | 31 .. 0 |
//	| 63     ....     0 | ⋯ | 63     ....     0 |
//	         z[3]         ⋯          z[0]
func reshapeDiagram(x, z specexpr.Vector) string {
	scalable := x.Scalable() || z.Scalable()
	if !scalable {
		xW, okXW := x.Width.(specexpr.Int)
		zW, okZW := z.Width.(specexpr.Int)
		if !okXW || !okZW || xW != zW {
			return ""
		}
	}

	// Order x and z by which is smaller and larger
	s, l := x, z
	sName, lName := "x", "z"
	if s.Elem.Bits > l.Elem.Bits {
		s, l = l, s
		sName, lName = lName, sName
	}

	R := int(l.Elem.Bits / s.Elem.Bits)

	// Table formatting helpers
	type table []string
	// vstack combines two tables vertically.
	vstack := func(tables ...table) table {
		return slices.Concat(tables...)
	}
	// hstack combines two tables horizontally.
	hstack := func(tables ...table) table {
		out := make(table, len(tables[0]))
		for row := range out {
			for _, table := range tables {
				out[row] += table[row]
			}
		}
		return out
	}
	repeat4 := func(x string) table { return table{x, x, x, x} }
	// format returns left+pad+center+pad+right, such that the length is at least width.
	format := func(left, center, right string, width int) string {
		pad := max(2, width-len(left)-len(center)-len(right)-2)
		// The (pad+1)/2 is because pad == pad/2 + (pad+1)/2 with truncating division.
		return " " + left + strings.Repeat(" ", (pad+1)/2) + center + strings.Repeat(" ", pad/2) + right + " "
	}

	// formatSmallCells formats the labels and boxes for small vector indexes [from, to].
	sBox := format(fmt.Sprintf("%d", s.Elem.Bits-1), "..", "0", 9)
	formatSmallCells := func(from, to int) table {
		var boxes, labels string
		for sIdx := from; sIdx >= to; sIdx-- {
			labels += format("", fmt.Sprintf("%s[%d]", sName, sIdx), "", len(sBox)) + " "
			boxes += sBox + "|"
		}
		return table{labels, boxes}
	}

	// formatCol constructs a full table column with small and large cells for a
	// single large vector index.
	formatCol := func(largeIdx int) table {
		high := (largeIdx+1)*R - 1
		low := largeIdx * R

		// Format the small cells that fan-out to this large cell, This gives us
		// metrics for formatting the large cell.
		var small table
		if R <= 4 {
			// There's enough room to show all of the small elements that form
			// one large element.
			small = formatSmallCells(high, low)
		} else {
			// The fan-out is too high. Show the left and right 2 small cells
			// and elide the middle.
			small = hstack(formatSmallCells(high, high-1), table{" ⋯  ", " ⋯ |"}, formatSmallCells(low+1, low))
		}

		// Format the large cell.
		width := utf8.RuneCountInString(small[0])
		large := table{
			format(fmt.Sprintf("%d", l.Elem.Bits-1), "....", "0", width-1) + "|",
			format("", fmt.Sprintf("%s[%d]", lName, largeIdx), "", width-1) + " "}
		// Stack small and large cells
		return vstack(small, large)
	}

	// Format columns for exactly two large vector elements.
	var left, right table
	cellSeps := table{" ", "|", "|", " "}
	prefix := cellSeps
	colSep := table{"", "", "", ""}
	if scalable {
		left, right = formatCol(1), formatCol(0)
		// For scalable vectors, we format large elements 0 and 1 and then show
		// it goes on forever to the left.
		prefix = hstack(repeat4("⋯ "), prefix)
	} else {
		leftIndex := int(l.Elems().(specexpr.Int)) - 1
		left, right = formatCol(leftIndex), formatCol(0)
		if leftIndex > 1 {
			// We're only showing the top and bottom large element, so show that
			// we're eliding the middle elements.
			colSep = hstack(repeat4(" ⋯ "), cellSeps)
		}
	}

	// Put it all together
	tab := hstack(prefix, left, colSep, right)

	// We built the smaller elements on top, so flip the table if they should be
	// on bottom.
	if x.Elem.Bits > z.Elem.Bits {
		slices.Reverse(tab)
	}

	// Finally, emit
	for i := range tab {
		tab[i] = strings.TrimRight(tab[i], " ")
	}
	return "\t  " + strings.Join(tab, "\n\t  ")
}

// newSpecTemplate parses a spec template string. If tmpl does not contain "{{",
// it is treated as a raw string literal without template overhead.
func newSpecTemplate(tmpl string) (specTemplate, error) {
	if !strings.Contains(tmpl, "{{") {
		return specTemplate{raw: tmpl, tmpl: nil}, nil
	}

	t, err := template.New("").Option("missingkey=error").Funcs(templateFuncs).Parse(tmpl)
	if err != nil {
		return specTemplate{}, err
	}
	return specTemplate{raw: tmpl, tmpl: t}, nil
}

func (s *specTemplate) expand(data any) (string, error) {
	if s.tmpl == nil {
		return s.raw, nil
	}
	var buf bytes.Buffer
	if err := s.tmpl.Execute(&buf, data); err != nil {
		return "", err
	}
	return buf.String(), nil
}

var reMultipleNewlines = regexp.MustCompile(`\n{3,}`)

func cleanDocNewlines(s string) string {
	s = reMultipleNewlines.ReplaceAllString(s, "\n\n")
	trimmed := strings.TrimRight(s, " \t\n")
	if trimmed == "" {
		return ""
	}
	return trimmed + "\n"
}

// expandNameAndDoc instantiates the API function name and doc comment for sFn
// using the solved variable bindings in b.
func (sFn *specFunc) expandNameAndDoc(ctx context, b *specexpr.Bindings) (name, doc string) {
	data := make(map[string]any)
	for v, val := range b.All() {
		data[string(v)] = val
		if s, ok := strings.CutPrefix(string(v), "$"); ok {
			data[s] = val
		}
	}

	var err error
	name, err = sFn.NameTmpl.expand(data)
	if err != nil {
		ctx.errorf("expanding function name template: %s", err)
		name = sFn.Name
	}

	data["Name"] = name
	doc, err = sFn.Doc.expand(data)
	if err != nil {
		ctx.errorf("expanding doc template: %s", err)
		doc = sFn.Doc.raw
	}

	doc = cleanDocNewlines(doc)

	if name != sFn.Name {
		doc = regexp.MustCompile(`\b`+regexp.QuoteMeta(sFn.Name)+`\b`).ReplaceAllLiteralString(doc, name)
	}

	return name, doc
}
