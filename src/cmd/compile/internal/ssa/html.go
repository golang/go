// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"bytes"
	"cmd/internal/src"
	"fmt"
	"html"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type HTMLWriter struct {
	Logger
	w    io.WriteCloser
	path string
}

func NewHTMLWriter(path string, logger Logger, funcname string) *HTMLWriter {
	out, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		logger.Fatalf(src.NoXPos, "%v", err)
	}
	pwd, err := os.Getwd()
	if err != nil {
		logger.Fatalf(src.NoXPos, "%v", err)
	}
	html := HTMLWriter{w: out, Logger: logger, path: filepath.Join(pwd, path)}
	html.start(funcname)
	return &html
}

func (w *HTMLWriter) start(name string) {
	if w == nil {
		return
	}
	w.WriteString("<html>")
	w.WriteString(`<head>
<meta http-equiv="Content-Type" content="text/html;charset=UTF-8">
<style>

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

.stats {
    font-size: 60%;
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
}

td > h2 {
    cursor: pointer;
    font-size: 120%;
}

td.collapsed {
    font-size: 12px;
    width: 12px;
    border: 0px;
    padding: 0;
    cursor: pointer;
    background: #fafafa;
}

td.collapsed  div {
     -moz-transform: rotate(-90.0deg);  /* FF3.5+ */
       -o-transform: rotate(-90.0deg);  /* Opera 10.5 */
  -webkit-transform: rotate(-90.0deg);  /* Saf3.1+, Chrome */
             filter:  progid:DXImageTransform.Microsoft.BasicImage(rotation=0.083);  /* IE6,IE7 */
         -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=0.083)"; /* IE8 */
         margin-top: 10.3em;
         margin-left: -10em;
         margin-right: -10em;
         text-align: right;
}

code, pre, .lines, .ast {
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

.lines {
    float: left;
    overflow: hidden;
    text-align: right;
}

.lines div {
    padding-right: 10px;
    color: gray;
}

div.line-number {
    font-size: 12px;
}

.ast {
    white-space: nowrap;
}

td.ssa-prog {
    width: 600px;
    word-wrap: break-word;
}

li {
    list-style-type: none;
}

li.ssa-long-value {
    text-indent: -2em;  /* indent wrapped lines */
}

li.ssa-value-list {
    display: inline;
}

li.ssa-start-block {
    padding: 0;
    margin: 0;
}

li.ssa-end-block {
    padding: 0;
    margin: 0;
}

ul.ssa-print-func {
    padding-left: 0;
}

dl.ssa-gen {
    padding-left: 0;
}

dt.ssa-prog-src {
    padding: 0;
    margin: 0;
    float: left;
    width: 4em;
}

dd.ssa-prog {
    padding: 0;
    margin-right: 0;
    margin-left: 4em;
}

.dead-value {
    color: gray;
}

.dead-block {
    opacity: 0.5;
}

.depcycle {
    font-style: italic;
}

.line-number {
    font-size: 11px;
}

.no-line-number {
    font-size: 11px;
    color: gray;
}

.highlight-aquamarine     { background-color: aquamarine; }
.highlight-coral          { background-color: coral; }
.highlight-lightpink      { background-color: lightpink; }
.highlight-lightsteelblue { background-color: lightsteelblue; }
.highlight-palegreen      { background-color: palegreen; }
.highlight-skyblue        { background-color: skyblue; }
.highlight-lightgray      { background-color: lightgray; }
.highlight-yellow         { background-color: yellow; }
.highlight-lime           { background-color: lime; }
.highlight-khaki          { background-color: khaki; }
.highlight-aqua           { background-color: aqua; }
.highlight-salmon         { background-color: salmon; }

.outline-blue           { outline: blue solid 2px; }
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

</style>

<script type="text/javascript">
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
    var ssaElemClicked = function(elem, event, selections, selected) {
        event.stopPropagation();

        // TODO: pushState with updated state and read it on page load,
        // so that state can survive across reloads

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

    var ssaValueClicked = function(event) {
        ssaElemClicked(this, event, highlights, highlighted);
    };

    var ssaBlockClicked = function(event) {
        ssaElemClicked(this, event, outlines, outlined);
    };

    var ssavalues = document.getElementsByClassName("ssa-value");
    for (var i = 0; i < ssavalues.length; i++) {
        ssavalues[i].addEventListener('click', ssaValueClicked);
    }

    var ssalongvalues = document.getElementsByClassName("ssa-long-value");
    for (var i = 0; i < ssalongvalues.length; i++) {
        // don't attach listeners to li nodes, just the spans they contain
        if (ssalongvalues[i].nodeName == "SPAN") {
            ssalongvalues[i].addEventListener('click', ssaValueClicked);
        }
    }

    var ssablocks = document.getElementsByClassName("ssa-block");
    for (var i = 0; i < ssablocks.length; i++) {
        ssablocks[i].addEventListener('click', ssaBlockClicked);
    }

    var lines = document.getElementsByClassName("line-number");
    for (var i = 0; i < lines.length; i++) {
        lines[i].addEventListener('click', ssaValueClicked);
    }

    // Contains phase names which are expanded by default. Other columns are collapsed.
    var expandedDefault = [
        "start",
        "deadcode",
        "opt",
        "lower",
        "late deadcode",
        "regalloc",
        "genssa",
    ];

    function toggler(phase) {
        return function() {
            toggle_cell(phase+'-col');
            toggle_cell(phase+'-exp');
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
    var td = document.getElementsByTagName("td");
    for (var i = 0; i < td.length; i++) {
        var id = td[i].id;
        var phase = id.substr(0, id.length-4);
        var show = expandedDefault.indexOf(phase) !== -1
        if (id.endsWith("-exp")) {
            var h2 = td[i].getElementsByTagName("h2");
            if (h2 && h2[0]) {
                h2[0].addEventListener('click', toggler(phase));
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
};

function toggle_visibility(id) {
    var e = document.getElementById(id);
    if (e.style.display == 'block') {
        e.style.display = 'none';
    } else {
        e.style.display = 'block';
    }
}
</script>

</head>`)
	w.WriteString("<body>")
	w.WriteString("<h1>")
	w.WriteString(html.EscapeString(name))
	w.WriteString("</h1>")
	w.WriteString(`
<a href="#" onclick="toggle_visibility('help');" id="helplink">help</a>
<div id="help">

<p>
Click on a value or block to toggle highlighting of that value/block
and its uses.  (Values and blocks are highlighted by ID, and IDs of
dead items may be reused, so not all highlights necessarily correspond
to the clicked item.)
</p>

<p>
Faded out values and blocks are dead code that has not been eliminated.
</p>

<p>
Values printed in italics have a dependency cycle.
</p>

</div>
`)
	w.WriteString("<table>")
	w.WriteString("<tr>")
}

func (w *HTMLWriter) Close() {
	if w == nil {
		return
	}
	io.WriteString(w.w, "</tr>")
	io.WriteString(w.w, "</table>")
	io.WriteString(w.w, "</body>")
	io.WriteString(w.w, "</html>")
	w.w.Close()
	fmt.Printf("dumped SSA to %v\n", w.path)
}

// WriteFunc writes f in a column headed by title.
// phase is used for collapsing columns and should be unique across the table.
func (w *HTMLWriter) WriteFunc(phase, title string, f *Func) {
	if w == nil {
		return // avoid generating HTML just to discard it
	}
	w.WriteColumn(phase, title, "", f.HTML())
	// TODO: Add visual representation of f's CFG.
}

// FuncLines contains source code for a function to be displayed
// in sources column.
type FuncLines struct {
	Filename    string
	StartLineno uint
	Lines       []string
}

// ByTopo sorts topologically: target function is on top,
// followed by inlined functions sorted by filename and line numbers.
type ByTopo []*FuncLines

func (x ByTopo) Len() int      { return len(x) }
func (x ByTopo) Swap(i, j int) { x[i], x[j] = x[j], x[i] }
func (x ByTopo) Less(i, j int) bool {
	a := x[i]
	b := x[j]
	if a.Filename == b.Filename {
		return a.StartLineno < b.StartLineno
	}
	return a.Filename < b.Filename
}

// WriteSources writes lines as source code in a column headed by title.
// phase is used for collapsing columns and should be unique across the table.
func (w *HTMLWriter) WriteSources(phase string, all []*FuncLines) {
	if w == nil {
		return // avoid generating HTML just to discard it
	}
	var buf bytes.Buffer
	fmt.Fprint(&buf, "<div class=\"lines\" style=\"width: 8%\">")
	filename := ""
	for _, fl := range all {
		fmt.Fprint(&buf, "<div>&nbsp;</div>")
		if filename != fl.Filename {
			fmt.Fprint(&buf, "<div>&nbsp;</div>")
			filename = fl.Filename
		}
		for i := range fl.Lines {
			ln := int(fl.StartLineno) + i
			fmt.Fprintf(&buf, "<div class=\"l%v line-number\">%v</div>", ln, ln)
		}
	}
	fmt.Fprint(&buf, "</div><div style=\"width: 92%\"><pre>")
	filename = ""
	for _, fl := range all {
		fmt.Fprint(&buf, "<div>&nbsp;</div>")
		if filename != fl.Filename {
			fmt.Fprintf(&buf, "<div><strong>%v</strong></div>", fl.Filename)
			filename = fl.Filename
		}
		for i, line := range fl.Lines {
			ln := int(fl.StartLineno) + i
			var escaped string
			if strings.TrimSpace(line) == "" {
				escaped = "&nbsp;"
			} else {
				escaped = html.EscapeString(line)
			}
			fmt.Fprintf(&buf, "<div class=\"l%v line-number\">%v</div>", ln, escaped)
		}
	}
	fmt.Fprint(&buf, "</pre></div>")
	w.WriteColumn(phase, phase, "allow-x-scroll", buf.String())
}

func (w *HTMLWriter) WriteAST(phase string, buf *bytes.Buffer) {
	if w == nil {
		return // avoid generating HTML just to discard it
	}
	lines := strings.Split(buf.String(), "\n")
	var out bytes.Buffer

	fmt.Fprint(&out, "<div>")
	for _, l := range lines {
		l = strings.TrimSpace(l)
		var escaped string
		var lineNo string
		if l == "" {
			escaped = "&nbsp;"
		} else {
			if strings.HasPrefix(l, "buildssa") {
				escaped = fmt.Sprintf("<b>%v</b>", l)
			} else {
				// Parse the line number from the format l(123).
				idx := strings.Index(l, " l(")
				if idx != -1 {
					subl := l[idx+3:]
					idxEnd := strings.Index(subl, ")")
					if idxEnd != -1 {
						if _, err := strconv.Atoi(subl[:idxEnd]); err == nil {
							lineNo = subl[:idxEnd]
						}
					}
				}
				escaped = html.EscapeString(l)
			}
		}
		if lineNo != "" {
			fmt.Fprintf(&out, "<div class=\"l%v line-number ast\">%v</div>", lineNo, escaped)
		} else {
			fmt.Fprintf(&out, "<div class=\"ast\">%v</div>", escaped)
		}
	}
	fmt.Fprint(&out, "</div>")
	w.WriteColumn(phase, phase, "allow-x-scroll", out.String())
}

// WriteColumn writes raw HTML in a column headed by title.
// It is intended for pre- and post-compilation log output.
func (w *HTMLWriter) WriteColumn(phase, title, class, html string) {
	if w == nil {
		return
	}
	id := strings.Replace(phase, " ", "-", -1)
	// collapsed column
	w.Printf("<td id=\"%v-col\" class=\"collapsed\"><div>%v</div></td>", id, phase)

	if class == "" {
		w.Printf("<td id=\"%v-exp\">", id)
	} else {
		w.Printf("<td id=\"%v-exp\" class=\"%v\">", id, class)
	}
	w.WriteString("<h2>" + title + "</h2>")
	w.WriteString(html)
	w.WriteString("</td>")
}

func (w *HTMLWriter) Printf(msg string, v ...interface{}) {
	if _, err := fmt.Fprintf(w.w, msg, v...); err != nil {
		w.Fatalf(src.NoXPos, "%v", err)
	}
}

func (w *HTMLWriter) WriteString(s string) {
	if _, err := io.WriteString(w.w, s); err != nil {
		w.Fatalf(src.NoXPos, "%v", err)
	}
}

func (v *Value) HTML() string {
	// TODO: Using the value ID as the class ignores the fact
	// that value IDs get recycled and that some values
	// are transmuted into other values.
	s := v.String()
	return fmt.Sprintf("<span class=\"%s ssa-value\">%s</span>", s, s)
}

func (v *Value) LongHTML() string {
	// TODO: Any intra-value formatting?
	// I'm wary of adding too much visual noise,
	// but a little bit might be valuable.
	// We already have visual noise in the form of punctuation
	// maybe we could replace some of that with formatting.
	s := fmt.Sprintf("<span class=\"%s ssa-long-value\">", v.String())

	linenumber := "<span class=\"no-line-number\">(?)</span>"
	if v.Pos.IsKnown() {
		linenumber = fmt.Sprintf("<span class=\"l%v line-number\">(%s)</span>", v.Pos.LineNumber(), v.Pos.LineNumberHTML())
	}

	s += fmt.Sprintf("%s %s = %s", v.HTML(), linenumber, v.Op.String())

	s += " &lt;" + html.EscapeString(v.Type.String()) + "&gt;"
	s += html.EscapeString(v.auxString())
	for _, a := range v.Args {
		s += fmt.Sprintf(" %s", a.HTML())
	}
	r := v.Block.Func.RegAlloc
	if int(v.ID) < len(r) && r[v.ID] != nil {
		s += " : " + html.EscapeString(r[v.ID].String())
	}
	var names []string
	for name, values := range v.Block.Func.NamedValues {
		for _, value := range values {
			if value == v {
				names = append(names, name.String())
				break // drop duplicates.
			}
		}
	}
	if len(names) != 0 {
		s += " (" + strings.Join(names, ", ") + ")"
	}

	s += "</span>"
	return s
}

func (b *Block) HTML() string {
	// TODO: Using the value ID as the class ignores the fact
	// that value IDs get recycled and that some values
	// are transmuted into other values.
	s := html.EscapeString(b.String())
	return fmt.Sprintf("<span class=\"%s ssa-block\">%s</span>", s, s)
}

func (b *Block) LongHTML() string {
	// TODO: improve this for HTML?
	s := fmt.Sprintf("<span class=\"%s ssa-block\">%s</span>", html.EscapeString(b.String()), html.EscapeString(b.Kind.String()))
	if b.Aux != nil {
		s += html.EscapeString(fmt.Sprintf(" {%v}", b.Aux))
	}
	if b.Control != nil {
		s += fmt.Sprintf(" %s", b.Control.HTML())
	}
	if len(b.Succs) > 0 {
		s += " &#8594;" // right arrow
		for _, e := range b.Succs {
			c := e.b
			s += " " + c.HTML()
		}
	}
	switch b.Likely {
	case BranchUnlikely:
		s += " (unlikely)"
	case BranchLikely:
		s += " (likely)"
	}
	if b.Pos.IsKnown() {
		// TODO does not begin to deal with the full complexity of line numbers.
		// Maybe we want a string/slice instead, of outer-inner when inlining.
		s += fmt.Sprintf(" <span class=\"l%v line-number\">(%s)</span>", b.Pos.LineNumber(), b.Pos.LineNumberHTML())
	}
	return s
}

func (f *Func) HTML() string {
	var buf bytes.Buffer
	fmt.Fprint(&buf, "<code>")
	p := htmlFuncPrinter{w: &buf}
	fprintFunc(p, f)

	// fprintFunc(&buf, f) // TODO: HTML, not text, <br /> for line breaks, etc.
	fmt.Fprint(&buf, "</code>")
	return buf.String()
}

type htmlFuncPrinter struct {
	w io.Writer
}

func (p htmlFuncPrinter) header(f *Func) {}

func (p htmlFuncPrinter) startBlock(b *Block, reachable bool) {
	// TODO: Make blocks collapsable?
	var dead string
	if !reachable {
		dead = "dead-block"
	}
	fmt.Fprintf(p.w, "<ul class=\"%s ssa-print-func %s\">", b, dead)
	fmt.Fprintf(p.w, "<li class=\"ssa-start-block\">%s:", b.HTML())
	if len(b.Preds) > 0 {
		io.WriteString(p.w, " &#8592;") // left arrow
		for _, e := range b.Preds {
			pred := e.b
			fmt.Fprintf(p.w, " %s", pred.HTML())
		}
	}
	io.WriteString(p.w, "</li>")
	if len(b.Values) > 0 { // start list of values
		io.WriteString(p.w, "<li class=\"ssa-value-list\">")
		io.WriteString(p.w, "<ul>")
	}
}

func (p htmlFuncPrinter) endBlock(b *Block) {
	if len(b.Values) > 0 { // end list of values
		io.WriteString(p.w, "</ul>")
		io.WriteString(p.w, "</li>")
	}
	io.WriteString(p.w, "<li class=\"ssa-end-block\">")
	fmt.Fprint(p.w, b.LongHTML())
	io.WriteString(p.w, "</li>")
	io.WriteString(p.w, "</ul>")
	// io.WriteString(p.w, "</span>")
}

func (p htmlFuncPrinter) value(v *Value, live bool) {
	var dead string
	if !live {
		dead = "dead-value"
	}
	fmt.Fprintf(p.w, "<li class=\"ssa-long-value %s\">", dead)
	fmt.Fprint(p.w, v.LongHTML())
	io.WriteString(p.w, "</li>")
}

func (p htmlFuncPrinter) startDepCycle() {
	fmt.Fprintln(p.w, "<span class=\"depcycle\">")
}

func (p htmlFuncPrinter) endDepCycle() {
	fmt.Fprintln(p.w, "</span>")
}

func (p htmlFuncPrinter) named(n LocalSlot, vals []*Value) {
	fmt.Fprintf(p.w, "<li>name %s: ", n)
	for _, val := range vals {
		fmt.Fprintf(p.w, "%s ", val.HTML())
	}
	fmt.Fprintf(p.w, "</li>")
}
