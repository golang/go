// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/* A little code to ease navigation of these documents.
 *
 * On window load we:
 *  + Bind search box hint placeholder show/hide events (bindSearchEvents)
 *  + Generate a table of contents (generateTOC)
 *  + Bind foldable sections (bindToggles)
 *  + Bind links to foldable sections (bindToggleLinks)
 */

(function() {
'use strict';

function bindSearchEvents() {

  var search = $('#search');
  if (search.length === 0) {
    return; // no search box
  }

  function clearInactive() {
    if (search.is('.inactive')) {
      search.val('');
      search.removeClass('inactive');
    }
  }

  function restoreInactive() {
    if (search.val() !== '') {
      return;
    }
    search.val(search.attr('placeholder'));
    search.addClass('inactive');
  }

  search.on('focus', clearInactive);
  search.on('blur', restoreInactive);

  restoreInactive();
}

/* Generates a table of contents: looks for h2 and h3 elements and generates
 * links. "Decorates" the element with id=="nav" with this table of contents.
 */
function generateTOC() {
  if ($('#manual-nav').length > 0) {
    return;
  }

  var nav = $('#nav');
  if (nav.length === 0) {
    return;
  }

  var toc_items = [];
  $(nav).nextAll('h2, h3').each(function() {
    var node = this;
    if (node.id == '')
      node.id = 'tmp_' + toc_items.length;
    var link = $('<a/>').attr('href', '#' + node.id).text($(node).text());
    var item;
    if ($(node).is('h2')) {
      item = $('<dt/>');
    } else { // h3
      item = $('<dd/>');
    }
    item.append(link);
    toc_items.push(item);
  });
  if (toc_items.length <= 1) {
    return;
  }

  var dl1 = $('<dl/>');
  var dl2 = $('<dl/>');

  var split_index = (toc_items.length / 2) + 1;
  if (split_index < 8) {
    split_index = toc_items.length;
  }
  for (var i = 0; i < split_index; i++) {
    dl1.append(toc_items[i]);
  }
  for (/* keep using i */; i < toc_items.length; i++) {
    dl2.append(toc_items[i]);
  }

  var tocTable = $('<table class="unruled"/>').appendTo(nav);
  var tocBody = $('<tbody/>').appendTo(tocTable);
  var tocRow = $('<tr/>').appendTo(tocBody);

  // 1st column
  $('<td class="first"/>').appendTo(tocRow).append(dl1);
  // 2nd column
  $('<td/>').appendTo(tocRow).append(dl2);
}

function bindToggle(el) {
  $('.toggleButton', el).click(function() {
    if ($(el).is('.toggle')) {
      $(el).addClass('toggleVisible').removeClass('toggle');
    } else {
      $(el).addClass('toggle').removeClass('toggleVisible');
    }
  });
}
function bindToggles(selector) {
  $(selector).each(function(i, el) {
    bindToggle(el);
  });
}

function bindToggleLink(el, prefix) {
  $(el).click(function() {
    var href = $(el).attr('href');
    var i = href.indexOf('#'+prefix);
    if (i < 0) {
      return;
    }
    var id = '#' + prefix + href.slice(i+1+prefix.length);
    if ($(id).is('.toggle')) {
      $(id).find('.toggleButton').first().click();
    }
  });
}
function bindToggleLinks(selector, prefix) {
  $(selector).each(function(i, el) {
    bindToggleLink(el, prefix);
  });
}

function setupDropdownPlayground() {
  if (!$('#page').is('.wide')) {
    return; // don't show on front page
  }
  var button = $('#playgroundButton');
  var div = $('#playground');
  var setup = false;
  button.toggle(function() {
    button.addClass('active');
    div.show();
    if (setup) {
      return;
    }
    setup = true;
    playground({
      'codeEl': $('.code', div),
      'outputEl': $('.output', div),
      'runEl': $('.run', div),
      'fmtEl': $('.fmt', div),
      'shareEl': $('.share', div),
      'shareRedirect': 'http://play.golang.org/p/'
    });
  },
  function() {
    button.removeClass('active');
    div.hide();
  });
  button.show();
  $('#menu').css('min-width', '+=60');
}

function setupInlinePlayground() {
	'use strict';
	// Set up playground when each element is toggled.
	$('div.play').each(function (i, el) {
		// Set up playground for this example.
		var setup = function() {
			var code = $('.code', el);
			playground({
				'codeEl':   code,
				'outputEl': $('.output', el),
				'runEl':    $('.run', el),
				'fmtEl':    $('.fmt', el),
				'shareEl':  $('.share', el),
				'shareRedirect': 'http://play.golang.org/p/'
			});

			// Make the code textarea resize to fit content.
			var resize = function() {
				code.height(0);
				var h = code[0].scrollHeight;
				code.height(h+20); // minimize bouncing.
				code.closest('.input').height(h);
			};
			code.on('keydown', resize);
			code.on('keyup', resize);
			code.keyup(); // resize now.
		};
		
		// If example already visible, set up playground now.
		if ($(el).is(':visible')) {
			setup();
			return;
		}

		// Otherwise, set up playground when example is expanded.
		var built = false;
		$(el).closest('.toggle').click(function() {
			// Only set up once.
			if (!built) {
				setup();
				built = true;
			}
		});
	});
}

// fixFocus tries to put focus to div#page so that keyboard navigation works.
function fixFocus() {
  var page = $('div#page');
  var topbar = $('div#topbar');
  page.css('outline', 0); // disable outline when focused
  page.attr('tabindex', -1); // and set tabindex so that it is focusable
  $(window).resize(function (evt) {
    // only focus page when the topbar is at fixed position (that is, it's in
    // front of page, and keyboard event will go to the former by default.)
    // by focusing page, keyboard event will go to page so that up/down arrow,
    // space, etc. will work as expected.
    if (topbar.css('position') == "fixed")
      page.focus();
  }).resize();
}

function toggleHash() {
    var hash = $(window.location.hash);
    if (hash.is('.toggle')) {
      hash.find('.toggleButton').first().click();
    }
}

function addPlusButtons() {
  var po = document.createElement('script');
  po.type = 'text/javascript';
  po.async = true;
  po.src = 'https://apis.google.com/js/platform.js';
  var s = document.getElementsByTagName('script')[0];
  s.parentNode.insertBefore(po, s);
}

$(document).ready(function() {
  bindSearchEvents();
  generateTOC();
  bindToggles(".toggle");
  bindToggles(".toggleVisible");
  bindToggleLinks(".exampleLink", "example_");
  bindToggleLinks(".overviewLink", "");
  bindToggleLinks(".examplesLink", "");
  bindToggleLinks(".indexLink", "");
  setupDropdownPlayground();
  setupInlinePlayground();
  fixFocus();
  toggleHash();
  addPlusButtons();

  // godoc.html defines window.initFuncs in the <head> tag, and root.html and
  // codewalk.js push their on-page-ready functions to the list.
  // We execute those functions here, to avoid loading jQuery until the page
  // content is loaded.
  for (var i = 0; i < window.initFuncs.length; i++) window.initFuncs[i]();
});

})();
