// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// opts is an object with these keys
// 	codeEl - code editor element 
// 	outputEl - program output element
// 	runEl - run button element
// 	shareEl - share button element (optional)
// 	shareURLEl - share URL text input element (optional)
// 	preCompile - callback to mutate request data before compiling
// 	postCompile - callback to read response data after compiling
//      simple - use plain textarea instead of CodeMirror.
function playground(opts) {
	var simple = opts['simple'];
	var code = $(opts['codeEl']);
	var editor;

	// autoindent helpers for simple mode.
	function insertTabs(n) {
		// find the selection start and end
		var start = code[0].selectionStart;
		var end   = code[0].selectionEnd;
		// split the textarea content into two, and insert n tabs
		var v = code[0].value;
		var u = v.substr(0, start);
		for (var i=0; i<n; i++) {
			u += "\t";
		}
		u += v.substr(end);
		// set revised content
		code[0].value = u;
		// reset caret position after inserted tabs
		code[0].selectionStart = start+n;
		code[0].selectionEnd = start+n;
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
			insertTabs(tabs, 1);
		}, 1);
	}

	function keyHandler(e) {
		if (simple && e.keyCode == 9) { // tab
			insertTabs(1);
			e.preventDefault();
			return false;
		}
		if (e.keyCode == 13) { // enter
			if (e.shiftKey) { // +shift
				run();
				e.preventDefault();
				return false;
			} else if (simple) {
				autoindent(e.target);
			}
		}
		return true;
	}
	if (simple) {
		code.unbind('keydown').bind('keydown', keyHandler);
	} else {
		editor = CodeMirror.fromTextArea(
			code[0],
			{
				lineNumbers: true,
				indentUnit: 8,
				indentWithTabs: true,
				onKeyEvent: function(editor, e) { keyHandler(e); }
			}
		);
	}
	var output = $(opts['outputEl']);

	function clearErrors() {
		if (!editor) {
			return;
		}
		var lines = editor.lineCount();
		for (var i = 0; i < lines; i++) {
			editor.setLineClass(i, null);
		}
	}
	function highlightErrors(text) {
		if (!editor) {
			return;
		}
		var errorRe = /[a-z]+\.go:([0-9]+): /g;
		var result;
		while ((result = errorRe.exec(text)) != null) {
			var line = result[1]*1-1;
			editor.setLineClass(line, "errLine")
		}
	}
	function body() {
		if (editor) {
			return editor.getValue();
		}
		return $(opts['codeEl']).val();
	}

	var seq = 0;
	function run() {
		clearErrors();
		output.removeClass("error").html(
			'<div class="loading">Waiting for remote server...</div>'
		);
		seq++;
		var cur = seq;
		var data = {"body": body()};
		if (opts['preCompile']) {
			opts['preCompile'](data);
		}
		$.ajax("/compile", {
			data: data,
			type: "POST",
			dataType: "json",
			success: function(data) {
				if (seq != cur) {
					return;
				}
				pre = $("<pre/>");
				output.empty().append(pre);
				if (opts['postCompile']) {
					opts['postCompile'](data);
				}
				if (!data) {
					return;
				}
				if (data.compile_errors != "") {
					pre.text(data.compile_errors);
					output.addClass("error");
					highlightErrors(data.compile_errors);
					return;
				}
				var out = ""+data.output;
				if (out.indexOf("IMAGE:") == 0) {
					var img = $("<img/>");
					var url = "data:image/png;base64,";
					url += out.substr(6)
					img.attr("src", url);
					output.empty().append(img);
					return;
				}
				pre.text(out);
			},
			error: function() {
				output.addClass("error").text(
					"Error communicating with remote server."
				);
			}
		});
	}
	$(opts['runEl']).click(run);

	if (opts['shareEl'] == null || opts['shareURLEl'] == null) {
		return editor;
	}

	function origin(href) {
		return (""+href).split("/").slice(0, 3).join("/");
	}

	var shareURL = $(opts['shareURLEl']).hide();
	var sharing = false;
	$(opts['shareEl']).click(function() {
		if (sharing) return;
		sharing = true;
		$.ajax("/share", {
			processData: false,
			data: body(),
			type: "POST",
			complete: function(xhr) {
				sharing = false;
				if (xhr.status != 200) {
					alert("Server error; try again.");
					return
				}
				var url = origin(window.location) + "/p/" +
					xhr.responseText;
				shareURL.show().val(url).focus().select();
			}
		});
	});

	return editor;
}
