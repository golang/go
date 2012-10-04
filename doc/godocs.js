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

$(document).ready(function() {
  bindSearchEvents();
  generateTOC();
  bindToggles(".toggle");
  bindToggles(".toggleVisible");
  bindToggleLinks(".exampleLink", "example_");
  bindToggleLinks(".overviewLink", "");
  bindToggleLinks(".examplesLink", "");
  bindToggleLinks(".indexLink", "");
});

})();
