// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/* A little code to ease navigation of these documents.
 *
 * On window load we:
 *  + Generate a table of contents (generateTOC)
 *  + Bind foldable sections (bindToggles)
 *  + Bind links to foldable sections (bindToggleLinks)
 */

(function() {
'use strict';

// Mobile-friendly topbar menu
$(function() {
  var menu = $('#menu');
  var menuButton = $('#menu-button');
  var menuButtonArrow = $('#menu-button-arrow');
  menuButton.click(function(event) {
    menu.toggleClass('menu-visible');
    menuButtonArrow.toggleClass('vertical-flip');
    event.preventDefault();
    return false;
  });
});

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
      item = $('<dd class="indent"/>');
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
    if ($(this).closest(".toggle, .toggleVisible")[0] != el) {
      // Only trigger the closest toggle header.
      return;
    }

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
      'shareRedirect': '//play.golang.org/p/'
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
				'shareRedirect': '//play.golang.org/p/'
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
  var id = window.location.hash.substring(1);
  // Open all of the toggles for a particular hash.
  var els = $(
    document.getElementById(id),
    $('a[name]').filter(function() {
      return $(this).attr('name') == id;
    })
  );

  while (els.length) {
    for (var i = 0; i < els.length; i++) {
      var el = $(els[i]);
      if (el.is('.toggle')) {
        el.find('.toggleButton').first().click();
      }
    }
    els = el.parent();
  }
}

function personalizeInstallInstructions() {
  var prefix = '?download=';
  var s = window.location.search;
  if (s.indexOf(prefix) != 0) {
    // No 'download' query string; detect "test" instructions from User Agent.
    if (navigator.platform.indexOf('Win') != -1) {
      $('.testUnix').hide();
      $('.testWindows').show();
    } else {
      $('.testUnix').show();
      $('.testWindows').hide();
    }
    return;
  }

  var filename = s.substr(prefix.length);
  var filenameRE = /^go1\.\d+(\.\d+)?([a-z0-9]+)?\.([a-z0-9]+)(-[a-z0-9]+)?(-osx10\.[68])?\.([a-z.]+)$/;
  $('.downloadFilename').text(filename);
  $('.hideFromDownload').hide();
  var m = filenameRE.exec(filename);
  if (!m) {
    // Can't interpret file name; bail.
    return;
  }

  var os = m[3];
  var ext = m[6];
  if (ext != 'tar.gz') {
    $('#tarballInstructions').hide();
  }
  if (os != 'darwin' || ext != 'pkg') {
    $('#darwinPackageInstructions').hide();
  }
  if (os != 'windows') {
    $('#windowsInstructions').hide();
    $('.testUnix').show();
    $('.testWindows').hide();
  } else {
    if (ext != 'msi') {
      $('#windowsInstallerInstructions').hide();
    }
    if (ext != 'zip') {
      $('#windowsZipInstructions').hide();
    }
    $('.testUnix').hide();
    $('.testWindows').show();
  }

  var download = "https://storage.googleapis.com/golang/" + filename;

  var message = $('<p class="downloading">'+
    'Your download should begin shortly. '+
    'If it does not, click <a>this link</a>.</p>');
  message.find('a').attr('href', download);
  message.insertAfter('#nav');

  window.location = download;
}

function updateVersionTags() {
  var v = window.goVersion;
  if (/^go[0-9.]+$/.test(v)) {
    $(".versionTag").empty().text(v);
    $(".whereTag").hide();
  }
}

function addPermalinks() {
  function addPermalink(source, parent) {
    var id = source.attr("id");
    if (id == "" || id.indexOf("tmp_") === 0) {
      // Auto-generated permalink.
      return;
    }
    if (parent.find("> .permalink").length) {
      // Already attached.
      return;
    }
    parent.append(" ").append($("<a class='permalink'>&#xb6;</a>").attr("href", "#" + id));
  }

  $("#page .container").find("h2[id], h3[id]").each(function() {
    var el = $(this);
    addPermalink(el, el);
  });

  $("#page .container").find("dl[id]").each(function() {
    var el = $(this);
    // Add the anchor to the "dt" element.
    addPermalink(el, el.find("> dt").first());
  });
}

$(document).ready(function() {
  generateTOC();
  addPermalinks();
  bindToggles(".toggle");
  bindToggles(".toggleVisible");
  bindToggleLinks(".exampleLink", "example_");
  bindToggleLinks(".overviewLink", "");
  bindToggleLinks(".examplesLink", "");
  bindToggleLinks(".indexLink", "");
  setupDropdownPlayground();
  setupInlinePlayground();
  fixFocus();
  setupTypeInfo();
  setupCallgraphs();
  toggleHash();
  personalizeInstallInstructions();
  updateVersionTags();

  // godoc.html defines window.initFuncs in the <head> tag, and root.html and
  // codewalk.js push their on-page-ready functions to the list.
  // We execute those functions here, to avoid loading jQuery until the page
  // content is loaded.
  for (var i = 0; i < window.initFuncs.length; i++) window.initFuncs[i]();
});

// -- analysis ---------------------------------------------------------

// escapeHTML returns HTML for s, with metacharacters quoted.
// It is safe for use in both elements and attributes
// (unlike the "set innerText, read innerHTML" trick).
function escapeHTML(s) {
    return s.replace(/&/g, '&amp;').
             replace(/\"/g, '&quot;').
             replace(/\'/g, '&#39;').
             replace(/</g, '&lt;').
             replace(/>/g, '&gt;');
}

// makeAnchor returns HTML for an <a> element, given an anchorJSON object.
function makeAnchor(json) {
  var html = escapeHTML(json.Text);
  if (json.Href != "") {
      html = "<a href='" + escapeHTML(json.Href) + "'>" + html + "</a>";
  }
  return html;
}

function showLowFrame(html) {
  var lowframe = document.getElementById('lowframe');
  lowframe.style.height = "200px";
  lowframe.innerHTML = "<p style='text-align: left;'>" + html + "</p>\n" +
      "<div onclick='hideLowFrame()' style='position: absolute; top: 0; right: 0; cursor: pointer;'>✘</div>"
};

document.hideLowFrame = function() {
  var lowframe = document.getElementById('lowframe');
  lowframe.style.height = "0px";
}

// onClickCallers is the onclick action for the 'func' tokens of a
// function declaration.
document.onClickCallers = function(index) {
  var data = document.ANALYSIS_DATA[index]
  if (data.Callers.length == 1 && data.Callers[0].Sites.length == 1) {
    document.location = data.Callers[0].Sites[0].Href; // jump to sole caller
    return;
  }

  var html = "Callers of <code>" + escapeHTML(data.Callee) + "</code>:<br/>\n";
  for (var i = 0; i < data.Callers.length; i++) {
    var caller = data.Callers[i];
    html += "<code>" + escapeHTML(caller.Func) + "</code>";
    var sites = caller.Sites;
    if (sites != null && sites.length > 0) {
      html += " at line ";
      for (var j = 0; j < sites.length; j++) {
        if (j > 0) {
          html += ", ";
        }
        html += "<code>" + makeAnchor(sites[j]) + "</code>";
      }
    }
    html += "<br/>\n";
  }
  showLowFrame(html);
};

// onClickCallees is the onclick action for the '(' token of a function call.
document.onClickCallees = function(index) {
  var data = document.ANALYSIS_DATA[index]
  if (data.Callees.length == 1) {
    document.location = data.Callees[0].Href; // jump to sole callee
    return;
  }

  var html = "Callees of this " + escapeHTML(data.Descr) + ":<br/>\n";
  for (var i = 0; i < data.Callees.length; i++) {
    html += "<code>" + makeAnchor(data.Callees[i]) + "</code><br/>\n";
  }
  showLowFrame(html);
};

// onClickTypeInfo is the onclick action for identifiers declaring a named type.
document.onClickTypeInfo = function(index) {
  var data = document.ANALYSIS_DATA[index];
  var html = "Type <code>" + data.Name + "</code>: " +
  "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<small>(size=" + data.Size + ", align=" + data.Align + ")</small><br/>\n";
  html += implementsHTML(data);
  html += methodsetHTML(data);
  showLowFrame(html);
};

// implementsHTML returns HTML for the implements relation of the
// specified TypeInfoJSON value.
function implementsHTML(info) {
  var html = "";
  if (info.ImplGroups != null) {
    for (var i = 0; i < info.ImplGroups.length; i++) {
      var group = info.ImplGroups[i];
      var x = "<code>" + escapeHTML(group.Descr) + "</code> ";
      for (var j = 0; j < group.Facts.length; j++) {
        var fact = group.Facts[j];
        var y = "<code>" + makeAnchor(fact.Other) + "</code>";
        if (fact.ByKind != null) {
          html += escapeHTML(fact.ByKind) + " type " + y + " implements " + x;
        } else {
          html += x + " implements " + y;
        }
        html += "<br/>\n";
      }
    }
  }
  return html;
}


// methodsetHTML returns HTML for the methodset of the specified
// TypeInfoJSON value.
function methodsetHTML(info) {
  var html = "";
  if (info.Methods != null) {
    for (var i = 0; i < info.Methods.length; i++) {
      html += "<code>" + makeAnchor(info.Methods[i]) + "</code><br/>\n";
    }
  }
  return html;
}

// onClickComm is the onclick action for channel "make" and "<-"
// send/receive tokens.
document.onClickComm = function(index) {
  var ops = document.ANALYSIS_DATA[index].Ops
  if (ops.length == 1) {
    document.location = ops[0].Op.Href; // jump to sole element
    return;
  }

  var html = "Operations on this channel:<br/>\n";
  for (var i = 0; i < ops.length; i++) {
    html += makeAnchor(ops[i].Op) + " by <code>" + escapeHTML(ops[i].Fn) + "</code><br/>\n";
  }
  if (ops.length == 0) {
    html += "(none)<br/>\n";
  }
  showLowFrame(html);
};

$(window).load(function() {
    // Scroll window so that first selection is visible.
    // (This means we don't need to emit id='L%d' spans for each line.)
    // TODO(adonovan): ideally, scroll it so that it's under the pointer,
    // but I don't know how to get the pointer y coordinate.
    var elts = document.getElementsByClassName("selection");
    if (elts.length > 0) {
	elts[0].scrollIntoView()
    }
});

// setupTypeInfo populates the "Implements" and "Method set" toggle for
// each type in the package doc.
function setupTypeInfo() {
  for (var i in document.ANALYSIS_DATA) {
    var data = document.ANALYSIS_DATA[i];

    var el = document.getElementById("implements-" + i);
    if (el != null) {
      // el != null => data is TypeInfoJSON.
      if (data.ImplGroups != null) {
        el.innerHTML = implementsHTML(data);
        el.parentNode.parentNode.style.display = "block";
      }
    }

    var el = document.getElementById("methodset-" + i);
    if (el != null) {
      // el != null => data is TypeInfoJSON.
      if (data.Methods != null) {
        el.innerHTML = methodsetHTML(data);
        el.parentNode.parentNode.style.display = "block";
      }
    }
  }
}

function setupCallgraphs() {
  if (document.CALLGRAPH == null) {
    return
  }
  document.getElementById("pkg-callgraph").style.display = "block";

  var treeviews = document.getElementsByClassName("treeview");
  for (var i = 0; i < treeviews.length; i++) {
    var tree = treeviews[i];
    if (tree.id == null || tree.id.indexOf("callgraph-") != 0) {
      continue;
    }
    var id = tree.id.substring("callgraph-".length);
    $(tree).treeview({collapsed: true, animated: "fast"});
    document.cgAddChildren(tree, tree, [id]);
    tree.parentNode.parentNode.style.display = "block";
  }
}

document.cgAddChildren = function(tree, ul, indices) {
  if (indices != null) {
    for (var i = 0; i < indices.length; i++) {
      var li = cgAddChild(tree, ul, document.CALLGRAPH[indices[i]]);
      if (i == indices.length - 1) {
        $(li).addClass("last");
      }
    }
  }
  $(tree).treeview({animated: "fast", add: ul});
}

// cgAddChild adds an <li> element for document.CALLGRAPH node cgn to
// the parent <ul> element ul. tree is the tree's root <ul> element.
function cgAddChild(tree, ul, cgn) {
   var li = document.createElement("li");
   ul.appendChild(li);
   li.className = "closed";

   var code = document.createElement("code");

   if (cgn.Callees != null) {
     $(li).addClass("expandable");

     // Event handlers and innerHTML updates don't play nicely together,
     // hence all this explicit DOM manipulation.
     var hitarea = document.createElement("div");
     hitarea.className = "hitarea expandable-hitarea";
     li.appendChild(hitarea);

     li.appendChild(code);

     var childUL = document.createElement("ul");
     li.appendChild(childUL);
     childUL.setAttribute('style', "display: none;");

     var onClick = function() {
       document.cgAddChildren(tree, childUL, cgn.Callees);
       hitarea.removeEventListener('click', onClick)
     };
     hitarea.addEventListener('click', onClick);

   } else {
     li.appendChild(code);
   }
   code.innerHTML += "&nbsp;" + makeAnchor(cgn.Func);
   return li
}

})();
