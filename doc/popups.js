// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

function godocs_bindPopups(data) {

  $('#content span').bind('mouseenter', function() {
    var id = $(this).attr('id');
    //var txt = $(this).text();
    if (typeof data[id] == 'undefined')
	return;
    var content = data[id];

    var $el = $('.popup', this);
    if (!$el.length) { // create it
      $el = $('<div class="popup"></div>');
      $el.prependTo(this).css($(this).offset()).text(content);
    }
  });
  $('#content span').bind('mouseleave', function() {
    $('.popup', this).remove();
  });

}
