// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"bufio"
	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"html"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
)

// An HTMLWriter dumps IR to multicolumn HTML, similar to what the
// ssa backend does for GOSSAFUNC.  This is not the format used for
// the ast column in GOSSAFUNC output.
type HTMLWriter struct {
	w             *BufferedWriterCloser
	Func          *Func
	canonIdMap    map[Node]int
	prevCanonId   int
	path          string
	prevHash      []byte
	pendingPhases []string
	pendingTitles []string
}

// BufferedWriterCloser is here to help avoid pre-buffering the whole
// rendered HTML in memory, which can cause problems for large inputs.
type BufferedWriterCloser struct {
	file io.Closer
	w    *bufio.Writer
}

func (b *BufferedWriterCloser) Write(p []byte) (n int, err error) {
	return b.w.Write(p)
}

func (b *BufferedWriterCloser) Close() error {
	b.w.Flush()
	b.w = nil
	return b.file.Close()
}

func NewBufferedWriterCloser(f io.WriteCloser) *BufferedWriterCloser {
	return &BufferedWriterCloser{file: f, w: bufio.NewWriter(f)}
}

func NewHTMLWriter(path string, f *Func, cfgMask string) *HTMLWriter {
	path = strings.ReplaceAll(path, "/", string(filepath.Separator))
	out, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		base.Fatalf("%v", err)
	}
	reportPath := path
	if !filepath.IsAbs(reportPath) {
		pwd, err := os.Getwd()
		if err != nil {
			base.Fatalf("%v", err)
		}
		reportPath = filepath.Join(pwd, path)
	}
	h := HTMLWriter{
		w:          NewBufferedWriterCloser(out),
		Func:       f,
		path:       reportPath,
		canonIdMap: make(map[Node]int),
	}
	h.start()
	return &h
}

// canonId assigns indices to nodes based on pointer identity.
// this helps ensure that output html files don't gratuitously
// differ from run to run.
func (h *HTMLWriter) canonId(n Node) int {
	if id := h.canonIdMap[n]; id > 0 {
		return id
	}
	h.prevCanonId++
	h.canonIdMap[n] = h.prevCanonId
	return h.prevCanonId
}

// Fatalf reports an error and exits.
func (w *HTMLWriter) Fatalf(msg string, args ...any) {
	base.FatalfAt(src.NoXPos, msg, args...)
}

const (
	RIGHT_ARROW = "\u25BA" // click-to-open (is closed)
	DOWN_ARROW  = "\u25BC" // click-to-close (is open)
)

func (w *HTMLWriter) start() {
	if w == nil {
		return
	}
	escName := html.EscapeString(PkgFuncName(w.Func))
	w.Print("<!DOCTYPE html>")
	w.Print("<html>")
	w.Printf(`<head>
<meta name="generator" content="AST display for %s">
<meta http-equiv="Content-Type" content="text/html;charset=UTF-8">
%s
%s
<title>AST display for %s</title>
</head>`, escName, CSS, JS, escName)
	w.Print("<body>")
	w.Print("<h1>")
	w.Print(html.EscapeString(w.Func.Sym().Name))
	w.Print("</h1>")
	w.Print(`
<a href="#" onclick="toggle_visibility('help');return false;" id="helplink">help</a>
<div id="help">

<p>
Click anywhere on a node (with "cell" cursor) to outline a node and all of its subtrees.
</p>
<p>
Click on a name (with "crosshair" cursor) to highlight every occurrence of a name.
(Note that all the name nodes are the same node, so those also all outline together).
</p>
<p>
Click on a file, line, or column (with "crosshair" cursor) to highlight positions
in that file, at that file:line, or at that file:line:column, respectively.<br>Inlined
locations are not treated as a single location, but as a sequence of locations that
can be independently highlighted.
</p>
<p>
Click on a ` + DOWN_ARROW + ` to collapse a subtree, or on a ` + RIGHT_ARROW + ` to expand a subtree.
</p>


</div>
<label for="dark-mode-button" style="margin-left: 15px; cursor: pointer;">darkmode</label>
<input type="checkbox" onclick="toggleDarkMode();" id="dark-mode-button" style="cursor: pointer" />
`)
	w.Print("<table>")
	w.Print("<tr>")
}

func (w *HTMLWriter) Close() {
	if w == nil {
		return
	}
	w.Print("</tr>")
	w.Print("</table>")
	w.Print("</body>")
	w.Print("</html>\n")
	w.w.Close()
	fmt.Fprintf(os.Stderr, "Writing html ast output for %s to %s\n", PkgFuncName(w.Func), w.path)
}

// WritePhase writes f in a column headed by title.
// phase is used for collapsing columns and should be unique across the table.
func (w *HTMLWriter) WritePhase(phase, title string) {
	if w == nil {
		return // avoid generating HTML just to discard it
	}
	w.pendingPhases = append(w.pendingPhases, phase)
	w.pendingTitles = append(w.pendingTitles, title)
	w.flushPhases()
}

// flushPhases collects any pending phases and titles, writes them to the html, and resets the pending slices.
func (w *HTMLWriter) flushPhases() {
	phaseLen := len(w.pendingPhases)
	if phaseLen == 0 {
		return
	}
	phases := strings.Join(w.pendingPhases, "  +  ")
	w.WriteMultiTitleColumn(
		phases,
		w.pendingTitles,
		"allow-x-scroll",
		w.FuncHTML(w.pendingPhases[phaseLen-1]),
	)
	w.pendingPhases = w.pendingPhases[:0]
	w.pendingTitles = w.pendingTitles[:0]
}

func (w *HTMLWriter) WriteMultiTitleColumn(phase string, titles []string, class string, writeContent func()) {
	if w == nil {
		return
	}
	id := strings.ReplaceAll(phase, " ", "-")
	// collapsed column
	w.Printf("<td id=\"%v-col\" class=\"collapsed\"><div>%v</div></td>", id, phase)

	if class == "" {
		w.Printf("<td id=\"%v-exp\">", id)
	} else {
		w.Printf("<td id=\"%v-exp\" class=\"%v\">", id, class)
	}
	for _, title := range titles {
		w.Print("<h2>" + title + "</h2>")
	}
	writeContent()
	w.Print("<div class=\"resizer\"></div>")
	w.Print("</td>\n")
}

func (w *HTMLWriter) Printf(msg string, v ...any) {
	if _, err := fmt.Fprintf(w.w, msg, v...); err != nil {
		w.Fatalf("%v", err)
	}
}

func (w *HTMLWriter) Print(s string) {
	if _, err := fmt.Fprint(w.w, s); err != nil {
		w.Fatalf("%v", err)
	}
}

func (w *HTMLWriter) indent(n int) {
	indent(w.w, n)
}

func (w *HTMLWriter) FuncHTML(phase string) func() {
	return func() {
		w.Print("<pre>") // use pre for formatting to preserve indentation
		w.dumpNodesHTML(w.Func.Body, 1)
		w.Print("</pre>")
	}
}

func (h *HTMLWriter) dumpNodesHTML(list Nodes, depth int) {
	if len(list) == 0 {
		h.Print(" <nil>")
		return
	}

	for _, n := range list {
		h.dumpNodeHTML(n, depth)
	}
}

// indent prints indentation to w.
func (h *HTMLWriter) indentForToggle(depth int, hasChildren bool) {
	h.Print("\n")
	if depth == 0 {
		return
	}
	for i := 0; i < depth-1; i++ {
		h.Print(".   ")
	}
	if hasChildren {
		h.Print(". ")
	} else {
		h.Print(".   ")
	}
}

func (h *HTMLWriter) dumpNodeHTML(n Node, depth int) {
	hasChildren := nodeHasChildren(n)
	h.indentForToggle(depth, hasChildren)

	if depth > 40 {
		h.Print("...")
		return
	}

	if n == nil {
		h.Print("NilIrNode")
		return
	}

	// For HTML, we want to wrap the node and its details in a span that can be highlighted
	// across all occurrences of the span in all columns, so it has to be linked to the node ID,
	// which is its address. Canonicalize the address to a counter so that repeated compiler
	// runs yield the same html.
	//
	// JS Equivalence logic:
	//   var c = elem.classList.item(0);
	//   var x = document.getElementsByClassName(c);
	//
	// Tag each class with its canonicalized index.

	h.Printf("<span class=\"n%d ir-node\">", h.canonId(n))
	defer h.Printf("</span>")

	if hasChildren {
		h.Print(`<span class="toggle" onclick="toggle_node(this)">` + DOWN_ARROW + `</span> `) // NOTE TRAILING SPACE after </span>!
	}

	if len(n.Init()) != 0 {
		h.Print(`<span class="node-body">`)
		h.Printf("%+v-init", n.Op())
		h.dumpNodesHTML(n.Init(), depth+1)
		h.indent(depth)
		h.Print(`</span>`)
	}

	switch n.Op() {
	default:
		h.Printf("%+v", n.Op())
		h.dumpNodeHeaderHTML(n)

	case OLITERAL:
		h.Printf("%+v-%v", n.Op(), html.EscapeString(fmt.Sprintf("%v", n.Val())))
		h.dumpNodeHeaderHTML(n)
		return

	case ONAME, ONONAME:
		if n.Sym() != nil {
			// Name highlighting:
			// Create a hash for the symbol name to use as a class
			// We use the same irValueClicked logic which uses the first class as the identifier
			name := fmt.Sprintf("%v", n.Sym())
			hash := sha256.Sum256([]byte(name))
			symID := "sym-" + hex.EncodeToString(hash[:6])
			h.Printf("%+v-<span class=\"%s variable-name\">%+v</span>", n.Op(), symID, html.EscapeString(name))
		} else {
			h.Printf("%+v", n.Op())
		}
		h.dumpNodeHeaderHTML(n)
		return

	case OLINKSYMOFFSET:
		n := n.(*LinksymOffsetExpr)
		h.Printf("%+v-%v", n.Op(), html.EscapeString(fmt.Sprintf("%v", n.Linksym)))
		if n.Offset_ != 0 {
			h.Printf("%+v", n.Offset_)
		}
		h.dumpNodeHeaderHTML(n)

	case OASOP:
		n := n.(*AssignOpStmt)
		h.Printf("%+v-%+v", n.Op(), n.AsOp)
		h.dumpNodeHeaderHTML(n)

	case OTYPE:
		h.Printf("%+v %+v", n.Op(), html.EscapeString(fmt.Sprintf("%v", n.Sym())))
		h.dumpNodeHeaderHTML(n)
		return

	case OCLOSURE:
		h.Printf("%+v", n.Op())
		h.dumpNodeHeaderHTML(n)

	case ODCLFUNC:
		n := n.(*Func)
		h.Printf("%+v", n.Op())
		h.dumpNodeHeaderHTML(n)
		if hasChildren {
			h.Print(`<span class="node-body">`)
			defer h.Print(`</span>`)
		}
		fn := n
		if len(fn.Dcl) > 0 {
			h.indent(depth)
			h.Printf("%+v-Dcl", n.Op())
			for _, dcl := range n.Dcl {
				h.dumpNodeHTML(dcl, depth+1)
			}
		}
		if len(fn.ClosureVars) > 0 {
			h.indent(depth)
			h.Printf("%+v-ClosureVars", n.Op())
			for _, cv := range fn.ClosureVars {
				h.dumpNodeHTML(cv, depth+1)
			}
		}
		if len(fn.Body) > 0 {
			h.indent(depth)
			h.Printf("%+v-body", n.Op())
			h.dumpNodesHTML(fn.Body, depth+1)
		}
		return
	}
	if hasChildren {
		h.Print(`<span class="node-body">`)
		defer h.Print(`</span>`)
	}

	v := reflect.ValueOf(n).Elem()
	t := reflect.TypeOf(n).Elem()
	nf := t.NumField()
	for i := 0; i < nf; i++ {
		tf := t.Field(i)
		vf := v.Field(i)
		if tf.PkgPath != "" {
			continue
		}
		switch tf.Type.Kind() {
		case reflect.Interface, reflect.Ptr, reflect.Slice:
			if vf.IsNil() {
				continue
			}
		}
		name := strings.TrimSuffix(tf.Name, "_")
		switch name {
		case "X", "Y", "Index", "Chan", "Value", "Call":
			name = ""
		}
		switch val := vf.Interface().(type) {
		case Node:
			if name != "" {
				h.indent(depth)
				h.Printf("%+v-%s", n.Op(), name)
			}
			h.dumpNodeHTML(val, depth+1)
		case Nodes:
			if len(val) == 0 {
				continue
			}
			if name != "" {
				h.indent(depth)
				h.Printf("%+v-%s", n.Op(), name)
			}
			h.dumpNodesHTML(val, depth+1)
		default:
			if vf.Kind() == reflect.Slice && vf.Type().Elem().Implements(nodeType) {
				if vf.Len() == 0 {
					continue
				}
				if name != "" {
					h.indent(depth)
					h.Printf("%+v-%s", n.Op(), name)
				}
				for i, n := 0, vf.Len(); i < n; i++ {
					h.dumpNodeHTML(vf.Index(i).Interface().(Node), depth+1)
				}
			}
		}
	}
}

func nodeHasChildren(n Node) bool {
	if n == nil {
		return false
	}
	if len(n.Init()) != 0 {
		return true
	}
	switch n.Op() {
	case OLITERAL, ONAME, ONONAME, OTYPE:
		return false
	case ODCLFUNC:
		n := n.(*Func)
		return len(n.Dcl) > 0 || len(n.ClosureVars) > 0 || len(n.Body) > 0
	}

	v := reflect.ValueOf(n).Elem()
	t := reflect.TypeOf(n).Elem()
	nf := t.NumField()
	for i := 0; i < nf; i++ {
		tf := t.Field(i)
		vf := v.Field(i)
		if tf.PkgPath != "" {
			continue
		}
		switch tf.Type.Kind() {
		case reflect.Interface, reflect.Ptr, reflect.Slice:
			if vf.IsNil() {
				continue
			}
		}
		switch val := vf.Interface().(type) {
		case Node:
			return true
		case Nodes:
			if len(val) > 0 {
				return true
			}
		default:
			if vf.Kind() == reflect.Slice && vf.Type().Elem().Implements(nodeType) {
				if vf.Len() > 0 {
					return true
				}
			}
		}
	}
	return false
}

func (h *HTMLWriter) dumpNodeHeaderHTML(n Node) {
	// print pointer to be able to see identical nodes
	if base.Debug.DumpPtrs != 0 {
		h.Printf(" p(%p)", n)
	}

	if base.Debug.DumpPtrs != 0 && n.Name() != nil && n.Name().Defn != nil {
		h.Printf(" defn(%p)", n.Name().Defn)
	}

	if base.Debug.DumpPtrs != 0 && n.Name() != nil && n.Name().Curfn != nil {
		h.Printf(" curfn(%p)", n.Name().Curfn)
	}
	if base.Debug.DumpPtrs != 0 && n.Name() != nil && n.Name().Outer != nil {
		h.Printf(" outer(%p)", n.Name().Outer)
	}

	if EscFmt != nil {
		if esc := EscFmt(n); esc != "" {
			h.Printf(" %s", html.EscapeString(esc))
		}
	}

	if n.Sym() != nil && n.Op() != ONAME && n.Op() != ONONAME && n.Op() != OTYPE {
		h.Printf(" %+v", html.EscapeString(fmt.Sprintf("%v", n.Sym())))
	}

	v := reflect.ValueOf(n).Elem()
	t := v.Type()
	nf := t.NumField()
	for i := 0; i < nf; i++ {
		tf := t.Field(i)
		if tf.PkgPath != "" {
			continue
		}
		k := tf.Type.Kind()
		if reflect.Bool <= k && k <= reflect.Complex128 {
			name := strings.TrimSuffix(tf.Name, "_")
			vf := v.Field(i)
			vfi := vf.Interface()
			if name == "Offset" && vfi == types.BADWIDTH || name != "Offset" && vf.IsZero() {
				continue
			}
			if vfi == true {
				h.Printf(" %s", name)
			} else {
				h.Printf(" %s:%+v", name, html.EscapeString(fmt.Sprintf("%v", vf.Interface())))
			}
		}
	}

	v = reflect.ValueOf(n)
	t = v.Type()
	nm := t.NumMethod()
	for i := 0; i < nm; i++ {
		tm := t.Method(i)
		if tm.PkgPath != "" {
			continue
		}
		m := v.Method(i)
		mt := m.Type()
		if mt.NumIn() == 0 && mt.NumOut() == 1 && mt.Out(0).Kind() == reflect.Bool {
			func() {
				defer func() { recover() }()
				if m.Call(nil)[0].Bool() {
					name := strings.TrimSuffix(tm.Name, "_")
					h.Printf(" %s", name)
				}
			}()
		}
	}

	if n.Op() == OCLOSURE {
		n := n.(*ClosureExpr)
		if fn := n.Func; fn != nil && fn.Nname.Sym() != nil {
			h.Printf(" fnName(%+v)", html.EscapeString(fmt.Sprintf("%v", fn.Nname.Sym())))
		}
	}

	if n.Type() != nil {
		if n.Op() == OTYPE {
			h.Printf(" type")
		}
		h.Printf(" %+v", html.EscapeString(fmt.Sprintf("%v", n.Type())))
	}
	if n.Typecheck() != 0 {
		h.Printf(" tc(%d)", n.Typecheck())
	}

	if n.Pos().IsKnown() {
		h.Print(" <span class=\"line-number\">")
		switch n.Pos().IsStmt() {
		case src.PosNotStmt:
			h.Print("_")
		case src.PosIsStmt:
			h.Print("+")
		}
		sep := ""
		base.Ctxt.AllPos(n.Pos(), func(pos src.Pos) {
			h.Print(sep)
			sep = " "
			// Hierarchical highlighting:
			// Click file -> highlight all ranges in this file
			// Click line -> highlight all ranges at this line (in this file)
			// Click col  -> highlight this specific range

			file := pos.Filename()
			// Create a hash for the filename to use as a class
			hash := sha256.Sum256([]byte(file))
			fileID := "loc-" + hex.EncodeToString(hash[:6])
			lineID := fmt.Sprintf("%s-L%d", fileID, pos.Line())
			colID := fmt.Sprintf("%s-C%d", lineID, pos.Col())

			// File part: triggers fileID
			h.Printf("<span class=\"%s line-number\">%s</span>:", fileID, html.EscapeString(filepath.Base(file)))
			// Line part: triggers lineID (and fileID via class list)
			h.Printf("<span class=\"%s %s line-number\">%d</span>:", lineID, fileID, pos.Line())
			// Col part: triggers colID (and lineID, fileID)
			h.Printf("<span class=\"%s %s %s line-number\">%d</span>", colID, lineID, fileID, pos.Col())
		})
		h.Print("</span>")
	}
}

const (
	CSS = `<style>

body {
    font-size: 14px;
    font-family: Arial, sans-serif;
}

h1 {
    font-size: 18px;
    display: inline-block;
    margin: 0 1em .5em 0;
}

#helplink {
    display: inline-block;
}

#help {
    display: none;
}

table {
    border: 1px solid black;
    table-layout: fixed;
    width: 300px;
}

th, td {
    border: 1px solid black;
    overflow: hidden;
    width: 400px;
    vertical-align: top;
    padding: 5px;
    position: relative;
}

.resizer {
    display: inline-block;
    background: transparent;
    width: 10px;
    height: 100%;
    position: absolute;
    right: 0;
    top: 0;
    cursor: col-resize;
    z-index: 100;
}

td > h2 {
    cursor: pointer;
    font-size: 120%;
    margin: 5px 0px 5px 0px;
}

td.collapsed {
    font-size: 12px;
    width: 12px;
    border: 1px solid white;
    padding: 2px;
    cursor: pointer;
    background: #fafafa;
}

td.collapsed div {
    text-align: right;
    transform: rotate(180deg);
    writing-mode: vertical-lr;
    white-space: pre;
}

pre {
    font-family: Menlo, monospace;
    font-size: 12px;
}

pre {
    -moz-tab-size: 4;
    -o-tab-size:   4;
    tab-size:      4;
}

.allow-x-scroll {
    overflow-x: scroll;
}

.ir-node {
    cursor: cell;
}

.variable-name {
    cursor: crosshair;
}

.line-number {
    font-size: 11px;
    cursor: crosshair;
}

body.darkmode {
    background-color: rgb(21, 21, 21);
    color: rgb(230, 255, 255);
    opacity: 100%;
}

td.darkmode {
    background-color: rgb(21, 21, 21);
    border: 1px solid gray;
}

body.darkmode table, th {
    border: 1px solid gray;
}

body.darkmode text {
    fill: white;
}

.highlight-aquamarine     { background-color: aquamarine; color: black; }
.highlight-coral          { background-color: coral; color: black; }
.highlight-lightpink      { background-color: lightpink; color: black; }
.highlight-lightsteelblue { background-color: lightsteelblue; color: black; }
.highlight-palegreen      { background-color: palegreen; color: black; }
.highlight-skyblue        { background-color: skyblue; color: black; }
.highlight-lightgray      { background-color: lightgray; color: black; }
.highlight-yellow         { background-color: yellow; color: black; }
.highlight-lime           { background-color: lime; color: black; }
.highlight-khaki          { background-color: khaki; color: black; }
.highlight-aqua           { background-color: aqua; color: black; }
.highlight-salmon         { background-color: salmon; color: black; }


.outline-blue           { outline: #2893ff solid 2px; }
.outline-red            { outline: red solid 2px; }
.outline-blueviolet     { outline: blueviolet solid 2px; }
.outline-darkolivegreen { outline: darkolivegreen solid 2px; }
.outline-fuchsia        { outline: fuchsia solid 2px; }
.outline-sienna         { outline: sienna solid 2px; }
.outline-gold           { outline: gold solid 2px; }
.outline-orangered      { outline: orangered solid 2px; }
.outline-teal           { outline: teal solid 2px; }
.outline-maroon         { outline: maroon solid 2px; }
.outline-black          { outline: black solid 2px; }

/* Capture alternative for outline-black and ellipse.outline-black when in dark mode */
body.darkmode .outline-black        { outline: gray solid 2px; }

.toggle {
    cursor: pointer;
    display: inline-block;
    text-align: center;
    user-select: none;
    font-size: 12px; // hand-tweaked
}

</style>
`

	JS = `<script type="text/javascript">

// Contains phase names which are expanded by default. Other columns are collapsed.
let expandedDefault = [
    "bloop",
    "loopvar",
    "escape",
    "slice",
    "walk",
];
if (history.state === null) {
    history.pushState({expandedDefault}, "", location.href);
}

// ordered list of all available highlight colors
var highlights = [
    "highlight-aquamarine",
    "highlight-coral",
    "highlight-lightpink",
    "highlight-lightsteelblue",
    "highlight-palegreen",
    "highlight-skyblue",
    "highlight-lightgray",
    "highlight-yellow",
    "highlight-lime",
    "highlight-khaki",
    "highlight-aqua",
    "highlight-salmon"
];

// state: which value is highlighted this color?
var highlighted = {};
for (var i = 0; i < highlights.length; i++) {
    highlighted[highlights[i]] = "";
}

// ordered list of all available outline colors
var outlines = [
    "outline-blue",
    "outline-red",
    "outline-blueviolet",
    "outline-darkolivegreen",
    "outline-fuchsia",
    "outline-sienna",
    "outline-gold",
    "outline-orangered",
    "outline-teal",
    "outline-maroon",
    "outline-black"
];

// state: which value is outlined this color?
var outlined = {};
for (var i = 0; i < outlines.length; i++) {
    outlined[outlines[i]] = "";
}

window.onload = function() {
    if (history.state !== null) {
        expandedDefault = history.state.expandedDefault;
    }
    if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
        toggleDarkMode();
        document.getElementById("dark-mode-button").checked = true;
    }

    var irElemClicked = function(elem, event, selections, selected) {
        event.stopPropagation();

        // find all values with the same name
        var c = elem.classList.item(0);
        var x = document.getElementsByClassName(c);

        // if selected, remove selections from all of them
        // otherwise, attempt to add

        var remove = "";
        for (var i = 0; i < selections.length; i++) {
            var color = selections[i];
            if (selected[color] == c) {
                remove = color;
                break;
            }
        }

        if (remove != "") {
            for (var i = 0; i < x.length; i++) {
                x[i].classList.remove(remove);
            }
            selected[remove] = "";
            return;
        }

        // we're adding a selection
        // find first available color
        var avail = "";
        for (var i = 0; i < selections.length; i++) {
            var color = selections[i];
            if (selected[color] == "") {
                avail = color;
                break;
            }
        }
        if (avail == "") {
            alert("out of selection colors; go add more");
            return;
        }

        // set that as the selection
        for (var i = 0; i < x.length; i++) {
            x[i].classList.add(avail);
        }
        selected[avail] = c;
    };

    var irValueClicked = function(event) {
        irElemClicked(this, event, highlights, highlighted);
    };

    var irTreeClicked = function(event) {
        irElemClicked(this, event, outlines, outlined);
    };

    var irValues = document.getElementsByClassName("ir-node");
    for (var i = 0; i < irValues.length; i++) {
        irValues[i].addEventListener('click', irTreeClicked);
    }

    var lines = document.getElementsByClassName("line-number");
    for (var i = 0; i < lines.length; i++) {
        lines[i].addEventListener('click', irValueClicked);
    }

    var variableNames = document.getElementsByClassName("variable-name");
    for (var i = 0; i < variableNames.length; i++) {
        variableNames[i].addEventListener('click', irValueClicked);
    }

    function toggler(phase) {
        return function() {
            toggle_cell(phase+'-col');
            toggle_cell(phase+'-exp');
            const i = expandedDefault.indexOf(phase);
            if (i !== -1) {
                expandedDefault.splice(i, 1);
            } else {
                expandedDefault.push(phase);
            }
            history.pushState({expandedDefault}, "", location.href);
        };
    }

    function toggle_cell(id) {
        var e = document.getElementById(id);
        if (e.style.display == 'table-cell') {
            e.style.display = 'none';
        } else {
            e.style.display = 'table-cell';
        }
    }

    // Go through all columns and collapse needed phases.
    const td = document.getElementsByTagName("td");
    for (let i = 0; i < td.length; i++) {
        const id = td[i].id;
        const phase = id.substr(0, id.length-4);
        let show = expandedDefault.indexOf(phase) !== -1

        // If show == false, check to see if this is a combined column (multiple phases).
        // If combined, check each of the phases to see if they are in our expandedDefaults.
        // If any are found, that entire combined column gets shown.
        if (!show) {
            const combined = phase.split('--+--');
            const len = combined.length;
            if (len > 1) {
                for (let i = 0; i < len; i++) {
                    const num = expandedDefault.indexOf(combined[i]);
                    if (num !== -1) {
                        expandedDefault.splice(num, 1);
                        if (expandedDefault.indexOf(phase) === -1) {
                            expandedDefault.push(phase);
                            show = true;
                        }
                    }
                }
            }
        }
        if (id.endsWith("-exp")) {
            const h2Els = td[i].getElementsByTagName("h2");
            const len = h2Els.length;
            if (len > 0) {
                for (let i = 0; i < len; i++) {
                    h2Els[i].addEventListener('click', toggler(phase));
                }
            }
        } else {
            td[i].addEventListener('click', toggler(phase));
        }
        if (id.endsWith("-col") && show || id.endsWith("-exp") && !show) {
            td[i].style.display = 'none';
            continue;
        }
        td[i].style.display = 'table-cell';
    }

    var resizers = document.getElementsByClassName("resizer");
    for (var i = 0; i < resizers.length; i++) {
        var resizer = resizers[i];
        resizer.addEventListener('mousedown', initDrag, false);
    }
};

var startX, startWidth, resizableCol;

function initDrag(e) {
    resizableCol = this.parentElement;
    startX = e.clientX;
    startWidth = parseInt(document.defaultView.getComputedStyle(resizableCol).width, 10);
    document.documentElement.addEventListener('mousemove', doDrag, false);
    document.documentElement.addEventListener('mouseup', stopDrag, false);
}

function doDrag(e) {
    resizableCol.style.width = (startWidth + e.clientX - startX) + 'px';
}

function stopDrag(e) {
    document.documentElement.removeEventListener('mousemove', doDrag, false);
    document.documentElement.removeEventListener('mouseup', stopDrag, false);
}

function toggle_visibility(id) {
    var e = document.getElementById(id);
    if (e.style.display == 'block') {
        e.style.display = 'none';
    } else {
        e.style.display = 'block';
    }
}

function toggleDarkMode() {
    document.body.classList.toggle('darkmode');

    // Collect all of the "collapsed" elements and apply dark mode on each collapsed column
    const collapsedEls = document.getElementsByClassName('collapsed');
    const len = collapsedEls.length;

    for (let i = 0; i < len; i++) {
        collapsedEls[i].classList.toggle('darkmode');
    }
}

function toggle_node(e) {
    event.stopPropagation();
    var parent = e.parentNode;
    var children = parent.children;
    for (var i = 0; i < children.length; i++) {
        if (children[i].classList.contains("node-body")) {
            if (children[i].style.display == "none") {
                children[i].style.display = "";
            } else {
                children[i].style.display = "none";
            }
        }
    }
    if (e.innerText == "` + RIGHT_ARROW + `") {
        e.innerText = "` + DOWN_ARROW + `";
    } else {
        e.innerText = "` + RIGHT_ARROW + `";
    }
}

</script>
`
)
