// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

function initPlayground(transport) {
	"use strict";

	function text(node) {
		var s = "";
		for (var i = 0; i < node.childNodes.length; i++) {
			var n = node.childNodes[i];
			if (n.nodeType === 1) {
				if (n.tagName === "BUTTON") continue
				if (n.tagName === "SPAN" && n.className === "number") continue;
				if (n.tagName === "DIV" || n.tagName == "BR") {
					s += "\n";
				}
				s += text(n);
				continue;
			}
			if (n.nodeType === 3) {
				s += n.nodeValue;
			}
		}
		return s.replace("\xA0", " "); // replace non-breaking spaces
	}

	function init(code) {
		var output = document.createElement('div');
		var outpre = document.createElement('pre');
		var running;

		if ($ && $(output).resizable) {
			$(output).resizable({
				handles: "n,w,nw",
				minHeight:	27,
				minWidth:	135,
				maxHeight: 608,
				maxWidth:	990
			});
		}

		function onKill() {
			if (running) running.Kill();
		}

		function onRun(e) {
			onKill();
			output.style.display = "block";
			outpre.innerHTML = "";
			run1.style.display = "none";
			var options = {Race: e.shiftKey};
			running = transport.Run(text(code), PlaygroundOutput(outpre), options);
		}

		function onClose() {
			onKill();
			output.style.display = "none";
			run1.style.display = "inline-block";
		}

		var run1 = document.createElement('button');
		run1.innerHTML = 'Run';
		run1.className = 'run';
		run1.addEventListener("click", onRun, false);
		var run2 = document.createElement('button');
		run2.className = 'run';
		run2.innerHTML = 'Run';
		run2.addEventListener("click", onRun, false);
		var kill = document.createElement('button');
		kill.className = 'kill';
		kill.innerHTML = 'Kill';
		kill.addEventListener("click", onKill, false);
		var close = document.createElement('button');
		close.className = 'close';
		close.innerHTML = 'Close';
		close.addEventListener("click", onClose, false);

		var button = document.createElement('div');
		button.classList.add('buttons');
		button.appendChild(run1);
		// Hack to simulate insertAfter
		code.parentNode.insertBefore(button, code.nextSibling);

		var buttons = document.createElement('div');
		buttons.classList.add('buttons');
		buttons.appendChild(run2);
		buttons.appendChild(kill);
		buttons.appendChild(close);

		output.classList.add('output');
		output.appendChild(buttons);
		output.appendChild(outpre);
		output.style.display = "none";
		code.parentNode.insertBefore(output, button.nextSibling);
	}

	var play = document.querySelectorAll('div.playground');
	for (var i = 0; i < play.length; i++) {
		init(play[i]);
	}
}

