// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"exec"
	"flag"
	"http"
	"io"
	"io/ioutil"
	"log"
	"os"
	"runtime"
	"strconv"
	"template"
)

var (
	httpListen = flag.String("http", "127.0.0.1:3999", "host:port to listen on")
	htmlOutput = flag.Bool("html", false, "render program output as HTML")
)

var (
	// a source of numbers, for naming temporary files
	uniq = make(chan int)
	// the architecture-identifying character of the tool chain, 5, 6, or 8
	archChar string
)

func main() {
	flag.Parse()

	// set archChar
	switch runtime.GOARCH {
	case "arm":
		archChar = "5"
	case "amd64":
		archChar = "6"
	case "386":
		archChar = "8"
	default:
		log.Fatalln("unrecognized GOARCH:", runtime.GOARCH)
	}

	// source of unique numbers
	go func() {
		for i := 0; ; i++ {
			uniq <- i
		}
	}()

	http.HandleFunc("/", FrontPage)
	http.HandleFunc("/compile", Compile)
	log.Fatal(http.ListenAndServe(*httpListen, nil))
}

// FrontPage is an HTTP handler that renders the goplay interface. 
// If a filename is supplied in the path component of the URI,
// its contents will be put in the interface's text area.
// Otherwise, the default "hello, world" program is displayed.
func FrontPage(w http.ResponseWriter, req *http.Request) {
	data, err := ioutil.ReadFile(req.URL.Path[1:])
	if err != nil {
		data = helloWorld
	}
	frontPage.Execute(w, data)
}

// Compile is an HTTP handler that reads Go source code from the request,
// compiles and links the code (returning any errors), runs the program, 
// and sends the program's output as the HTTP response.
func Compile(w http.ResponseWriter, req *http.Request) {
	// x is the base name for .go, .6, executable files
	x := os.TempDir() + "/compile" + strconv.Itoa(<-uniq)
	src := x + ".go"
	obj := x + "." + archChar
	bin := x
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}

	// write request Body to x.go
	f, err := os.Create(src)
	if err != nil {
		error_(w, nil, err)
		return
	}
	defer os.Remove(src)
	defer f.Close()
	_, err = io.Copy(f, req.Body)
	if err != nil {
		error_(w, nil, err)
		return
	}
	f.Close()

	// build x.go, creating x.6
	out, err := run(archChar+"g", "-o", obj, src)
	defer os.Remove(obj)
	if err != nil {
		error_(w, out, err)
		return
	}

	// link x.6, creating x (the program binary)
	out, err = run(archChar+"l", "-o", bin, obj)
	defer os.Remove(bin)
	if err != nil {
		error_(w, out, err)
		return
	}

	// run x
	out, err = run(bin)
	if err != nil {
		error_(w, out, err)
	}

	// write the output of x as the http response
	if *htmlOutput {
		w.Write(out)
	} else {
		output.Execute(w, out)
	}
}

// error writes compile, link, or runtime errors to the HTTP connection.
// The JavaScript interface uses the 404 status code to identify the error.
func error_(w http.ResponseWriter, out []byte, err error) {
	w.WriteHeader(404)
	if out != nil {
		output.Execute(w, out)
	} else {
		output.Execute(w, err.Error())
	}
}

// run executes the specified command and returns its output and an error.
func run(cmd ...string) ([]byte, error) {
	return exec.Command(cmd[0], cmd[1:]...).CombinedOutput()
}

var frontPage = template.Must(template.New("frontPage").Parse(frontPageText)) // HTML template
var output = template.Must(template.New("output").Parse(outputText))          // HTML template

var outputText = `<pre>{{printf "%s" . |html}}</pre>`

var frontPageText = `<!doctype html>
<html>
<head>
<style>
pre, textarea {
	font-family: Monaco, 'Courier New', 'DejaVu Sans Mono', 'Bitstream Vera Sans Mono', monospace;
	font-size: 100%;
}
.hints {
	font-size: 0.8em;
	text-align: right;
}
#edit, #output, #errors { width: 100%; text-align: left; }
#edit { height: 500px; }
#output { color: #00c; }
#errors { color: #c00; }
</style>
<script>

function insertTabs(n) {
	// find the selection start and end
	var cont  = document.getElementById("edit");
	var start = cont.selectionStart;
	var end   = cont.selectionEnd;
	// split the textarea content into two, and insert n tabs
	var v = cont.value;
	var u = v.substr(0, start);
	for (var i=0; i<n; i++) {
		u += "\t";
	}
	u += v.substr(end);
	// set revised content
	cont.value = u;
	// reset caret position after inserted tabs
	cont.selectionStart = start+n;
	cont.selectionEnd = start+n;
}

function autoindent(el) {
	var curpos = el.selectionStart;
	var tabs = 0;
	while (curpos > 0) {
		curpos--;
		if (el.value[curpos] == "\t") {
			tabs++;
		} else if (tabs > 0 || el.value[curpos] == "\n") {
			break;
		}
	}
	setTimeout(function() {
		insertTabs(tabs);
	}, 1);
}

function keyHandler(event) {
	var e = window.event || event;
	if (e.keyCode == 9) { // tab
		insertTabs(1);
		e.preventDefault();
		return false;
	}
	if (e.keyCode == 13) { // enter
		if (e.shiftKey) { // +shift
			compile(e.target);
			e.preventDefault();
			return false;
		} else {
			autoindent(e.target);
		}
	}
	return true;
}

var xmlreq;

function autocompile() {
	if(!document.getElementById("autocompile").checked) {
		return;
	}
	compile();
}

function compile() {
	var prog = document.getElementById("edit").value;
	var req = new XMLHttpRequest();
	xmlreq = req;
	req.onreadystatechange = compileUpdate;
	req.open("POST", "/compile", true);
	req.setRequestHeader("Content-Type", "text/plain; charset=utf-8");
	req.send(prog);	
}

function compileUpdate() {
	var req = xmlreq;
	if(!req || req.readyState != 4) {
		return;
	}
	if(req.status == 200) {
		document.getElementById("output").innerHTML = req.responseText;
		document.getElementById("errors").innerHTML = "";
	} else {
		document.getElementById("errors").innerHTML = req.responseText;
		document.getElementById("output").innerHTML = "";
	}
}
</script>
</head>
<body>
<table width="100%"><tr><td width="60%" valign="top">
<textarea autofocus="true" id="edit" spellcheck="false" onkeydown="keyHandler(event);" onkeyup="autocompile();">{{printf "%s" . |html}}</textarea>
<div class="hints">
(Shift-Enter to compile and run.)&nbsp;&nbsp;&nbsp;&nbsp;
<input type="checkbox" id="autocompile" value="checked" /> Compile and run after each keystroke
</div>
<td width="3%">
<td width="27%" align="right" valign="top">
<div id="output"></div>
</table>
<div id="errors"></div>
</body>
</html>
`

var helloWorld = []byte(`package main

import "fmt"

func main() {
	fmt.Println("hello, world")
}
`)
