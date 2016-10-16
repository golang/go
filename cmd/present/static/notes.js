// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Store child window object which will display slides with notes
var notesWindow = null;

var isParentWindow = window.parent == window;

// When parent window closes, clear storage and close child window
if (isParentWindow) {
  window.onbeforeunload = function() {
    localStorage.clear();
    if (notesWindow) notesWindow.close();
  }
};

function toggleNotesWindow() {
  if (!isParentWindow) return;
  if (notesWindow) {
    notesWindow.close();
    notesWindow = null;
    return;
  }

  initNotes();
};

function initNotes() {
  notesWindow = window.open('', '', 'width=1000,height=700');
  var w = notesWindow;
  var slidesUrl = window.location.href;

  var curSlide = parseInt(localStorage.getItem('destSlide'), 10);
  var formattedNotes = '';
  var section = sections[curSlide - 1];
  // curSlide is 0 when initialized from the first page of slides.
  // Check if section is valid before retrieving Notes.
  if (section) {
    formattedNotes = formatNotes(section.Notes);
  } else if (curSlide == 0) {
    formattedNotes = formatNotes(titleNotes);
  }

  // Hack to apply css. Requires existing html on notesWindow.
  w.document.write("<div style='display:none;'></div>");

  w.document.title = window.document.title;

  var slides = w.document.createElement('iframe');
  slides.id = 'presenter-slides';
  slides.src = slidesUrl;
  w.document.body.appendChild(slides);
  // setTimeout needed for Firefox
  setTimeout(function() {
    slides.focus();
  }, 100);

  var notes = w.document.createElement('div');
  notes.id = 'presenter-notes';
  notes.innerHTML = formattedNotes;
  w.document.body.appendChild(notes);

  w.document.close();

  function addPresenterNotesStyle() {
    var el = w.document.createElement('link');
    el.rel = 'stylesheet';
    el.type = 'text/css';
    el.href = PERMANENT_URL_PREFIX + 'notes.css';
    w.document.body.appendChild(el);
    w.document.querySelector('head').appendChild(el);
  }

  addPresenterNotesStyle();

  // Add listener on notesWindow to update notes when triggered from
  // parent window
  w.addEventListener('storage', updateNotes, false);
};

function formatNotes(notes) {
  var formattedNotes = '';
  if (notes) {
    for (var i = 0; i < notes.length; i++) {
      formattedNotes = formattedNotes + '<p>' + notes[i] + '</p>';
    }
  }
  return formattedNotes;
};

function updateNotes() {
  // When triggered from parent window, notesWindow is null
  // The storage event listener on notesWindow will update notes
  if (!notesWindow) return;
  var destSlide = parseInt(localStorage.getItem('destSlide'), 10);
  var section = sections[destSlide - 1];
  var el = notesWindow.document.getElementById('presenter-notes');

  if (!el) return;

  if (section && section.Notes) {
    el.innerHTML = formatNotes(section.Notes);
  } else if (destSlide == 0) {
    el.innerHTML = formatNotes(titleNotes);
  }  else {
    el.innerHTML = '';
  }
};

/* Playground syncing */

// When presenter notes are enabled, playground click handlers are
// stored here to sync click events on the correct playground
var playgroundHandlers = {onRun: [], onKill: [], onClose: []};

function updatePlay(e) {
	var i = localStorage.getItem('play-index');

	switch (e.key) {
		case 'play-index':
			return;
		case 'play-action':
			// Sync 'run', 'kill', 'close' actions
			var action = localStorage.getItem('play-action');
			playgroundHandlers[action][i](e);
			return;
		case 'play-code':
			// Sync code editing
			var play = document.querySelectorAll('div.playground')[i];
			play.innerHTML = localStorage.getItem('play-code');
			return;
		case 'output-style':
			// Sync resizing of playground output
			var out = document.querySelectorAll('.output')[i];
			out.style = localStorage.getItem('output-style');
			return;
	}
};

// Reset 'run', 'kill', 'close' storage items when synced
// so that successive actions can be synced correctly
function updatePlayStorage(action, index, e) {
	localStorage.setItem('play-index', index);

	if (localStorage.getItem('play-action') === action) {
		// We're the receiving window, and the message has been received
		localStorage.removeItem('play-action');
	} else {
		// We're the triggering window, send the message
		localStorage.setItem('play-action', action);
	}

	if (action === 'onRun') {
		if (localStorage.getItem('play-shiftKey') === 'true') {
			localStorage.removeItem('play-shiftKey');
		} else if (e.shiftKey) {
			localStorage.setItem('play-shiftKey', e.shiftKey);
		}
	}
};
