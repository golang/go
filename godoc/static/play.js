// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

function initPlayground(transport) {
  'use strict';

  function text(node) {
    var s = '';
    for (var i = 0; i < node.childNodes.length; i++) {
      var n = node.childNodes[i];
      if (n.nodeType === 1) {
        if (n.tagName === 'BUTTON') continue;
        if (n.tagName === 'SPAN' && n.className === 'number') continue;
        if (n.tagName === 'DIV' || n.tagName == 'BR') {
          s += '\n';
        }
        s += text(n);
        continue;
      }
      if (n.nodeType === 3) {
        s += n.nodeValue;
      }
    }
    return s.replace('\xA0', ' '); // replace non-breaking spaces
  }

  // When presenter notes are enabled, the index passed
  // here will identify the playground to be synced
  function init(code, index) {
    var output = document.createElement('div');
    var outpre = document.createElement('pre');
    var running;

    if ($ && $(output).resizable) {
      $(output).resizable({
        handles: 'n,w,nw',
        minHeight: 27,
        minWidth: 135,
        maxHeight: 608,
        maxWidth: 990,
      });
    }

    function onKill() {
      if (running) running.Kill();
      if (window.notesEnabled) updatePlayStorage('onKill', index);
    }

    function onRun(e) {
      var sk = e.shiftKey || localStorage.getItem('play-shiftKey') === 'true';
      if (running) running.Kill();
      output.style.display = 'block';
      outpre.textContent = '';
      run1.style.display = 'none';
      var options = { Race: sk };
      running = transport.Run(text(code), PlaygroundOutput(outpre), options);
      if (window.notesEnabled) updatePlayStorage('onRun', index, e);
    }

    function onClose() {
      if (running) running.Kill();
      output.style.display = 'none';
      run1.style.display = 'inline-block';
      if (window.notesEnabled) updatePlayStorage('onClose', index);
    }

    if (window.notesEnabled) {
      playgroundHandlers.onRun.push(onRun);
      playgroundHandlers.onClose.push(onClose);
      playgroundHandlers.onKill.push(onKill);
    }

    var run1 = document.createElement('button');
    run1.textContent = 'Run';
    run1.className = 'run';
    run1.addEventListener('click', onRun, false);
    var run2 = document.createElement('button');
    run2.className = 'run';
    run2.textContent = 'Run';
    run2.addEventListener('click', onRun, false);
    var kill = document.createElement('button');
    kill.className = 'kill';
    kill.textContent = 'Kill';
    kill.addEventListener('click', onKill, false);
    var close = document.createElement('button');
    close.className = 'close';
    close.textContent = 'Close';
    close.addEventListener('click', onClose, false);

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
    output.style.display = 'none';
    code.parentNode.insertBefore(output, button.nextSibling);
  }

  var play = document.querySelectorAll('div.playground');
  for (var i = 0; i < play.length; i++) {
    init(play[i], i);
  }
}
