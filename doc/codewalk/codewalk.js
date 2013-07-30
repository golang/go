// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/**
 * A class to hold information about the Codewalk Viewer.
 * @param {jQuery} context The top element in whose context the viewer should
 *     operate.  It will not touch any elements above this one.
 * @constructor
 */
 var CodewalkViewer = function(context) {
  this.context = context;

  /**
   * The div that contains all of the comments and their controls.
   */
  this.commentColumn = this.context.find('#comment-column');

  /**
   * The div that contains the comments proper.
   */
  this.commentArea = this.context.find('#comment-area');

  /**
   * The div that wraps the iframe with the code, as well as the drop down menu
   * listing the different files.
   * @type {jQuery}
   */
  this.codeColumn = this.context.find('#code-column');

  /**
   * The div that contains the code but excludes the options strip.
   * @type {jQuery}
   */
  this.codeArea = this.context.find('#code-area');

  /**
   * The iframe that holds the code (from Sourcerer).
   * @type {jQuery}
   */
  this.codeDisplay = this.context.find('#code-display');

  /**
   * The overlaid div used as a grab handle for sizing the code/comment panes.
   * @type {jQuery}
   */
  this.sizer = this.context.find('#sizer');

  /**
   * The full-screen overlay that ensures we don't lose track of the mouse
   * while dragging.
   * @type {jQuery}
   */
  this.overlay = this.context.find('#overlay');

  /**
   * The hidden input field that we use to hold the focus so that we can detect
   * shortcut keypresses.
   * @type {jQuery}
   */
  this.shortcutInput = this.context.find('#shortcut-input');

  /**
   * The last comment that was selected.
   * @type {jQuery}
   */
  this.lastSelected = null;
};

/**
 * Minimum width of the comments or code pane, in pixels.
 * @type {number}
 */
CodewalkViewer.MIN_PANE_WIDTH = 200;

/**
 * Navigate the code iframe to the given url and update the code popout link.
 * @param {string} url The target URL.
 * @param {Object} opt_window Window dependency injection for testing only.
 */
CodewalkViewer.prototype.navigateToCode = function(url, opt_window) {
  if (!opt_window) opt_window = window;
  // Each iframe is represented by two distinct objects in the DOM:  an iframe
  // object and a window object.  These do not expose the same capabilities.
  // Here we need to get the window representation to get the location member,
  // so we access it directly through window[] since jQuery returns the iframe
  // representation.
  // We replace location rather than set so as not to create a history for code
  // navigation.
  opt_window['code-display'].location.replace(url);
  var k = url.indexOf('&');
  if (k != -1) url = url.slice(0, k);
  k = url.indexOf('fileprint=');
  if (k != -1) url = url.slice(k+10, url.length);
  this.context.find('#code-popout-link').attr('href', url);
};

/**
 * Selects the first comment from the list and forces a refresh of the code
 * view.
 */
CodewalkViewer.prototype.selectFirstComment = function() {
  // TODO(rsc): handle case where there are no comments
  var firstSourcererLink = this.context.find('.comment:first');
  this.changeSelectedComment(firstSourcererLink);
};

/**
 * Sets the target on all links nested inside comments to be _blank.
 */
CodewalkViewer.prototype.targetCommentLinksAtBlank = function() {
  this.context.find('.comment a[href], #description a[href]').each(function() {
    if (!this.target) this.target = '_blank';
  });
};

/**
 * Installs event handlers for all the events we care about.
 */
CodewalkViewer.prototype.installEventHandlers = function() {
  var self = this;

  this.context.find('.comment')
      .click(function(event) {
        if (jQuery(event.target).is('a[href]')) return true;
        self.changeSelectedComment(jQuery(this));
        return false;
      });

  this.context.find('#code-selector')
      .change(function() {self.navigateToCode(jQuery(this).val());});

  this.context.find('#description-table .quote-feet.setting')
      .click(function() {self.toggleDescription(jQuery(this)); return false;});

  this.sizer
      .mousedown(function(ev) {self.startSizerDrag(ev); return false;});
  this.overlay
      .mouseup(function(ev) {self.endSizerDrag(ev); return false;})
      .mousemove(function(ev) {self.handleSizerDrag(ev); return false;});

  this.context.find('#prev-comment')
      .click(function() {
          self.changeSelectedComment(self.lastSelected.prev()); return false;
      });

  this.context.find('#next-comment')
      .click(function() {
          self.changeSelectedComment(self.lastSelected.next()); return false;
      });

  // Workaround for Firefox 2 and 3, which steal focus from the main document
  // whenever the iframe content is (re)loaded.  The input field is not shown,
  // but is a way for us to bring focus back to a place where we can detect
  // keypresses.
  this.context.find('#code-display')
      .load(function(ev) {self.shortcutInput.focus();});

  jQuery(document).keypress(function(ev) {
    switch(ev.which) {
      case 110:  // 'n'
          self.changeSelectedComment(self.lastSelected.next());
          return false;
      case 112:  // 'p'
          self.changeSelectedComment(self.lastSelected.prev());
          return false;
      default:  // ignore
    }
  });

  window.onresize = function() {self.updateHeight();};
};

/**
 * Starts dragging the pane sizer.
 * @param {Object} ev The mousedown event that started us dragging.
 */
CodewalkViewer.prototype.startSizerDrag = function(ev) {
  this.initialCodeWidth = this.codeColumn.width();
  this.initialCommentsWidth = this.commentColumn.width();
  this.initialMouseX = ev.pageX;
  this.overlay.show();
};

/**
 * Handles dragging the pane sizer.
 * @param {Object} ev The mousemove event updating dragging position.
 */
CodewalkViewer.prototype.handleSizerDrag = function(ev) {
  var delta = ev.pageX - this.initialMouseX;
  if (this.codeColumn.is('.right')) delta = -delta;
  var proposedCodeWidth = this.initialCodeWidth + delta;
  var proposedCommentWidth = this.initialCommentsWidth - delta;
  var mw = CodewalkViewer.MIN_PANE_WIDTH;
  if (proposedCodeWidth < mw) delta = mw - this.initialCodeWidth;
  if (proposedCommentWidth < mw) delta = this.initialCommentsWidth - mw;
  proposedCodeWidth = this.initialCodeWidth + delta;
  proposedCommentWidth = this.initialCommentsWidth - delta;
  // If window is too small, don't even try to resize.
  if (proposedCodeWidth < mw || proposedCommentWidth < mw) return;
  this.codeColumn.width(proposedCodeWidth);
  this.commentColumn.width(proposedCommentWidth);
  this.options.codeWidth = parseInt(
      this.codeColumn.width() /
      (this.codeColumn.width() + this.commentColumn.width()) * 100);
  this.context.find('#code-column-width').text(this.options.codeWidth + '%');
};

/**
 * Ends dragging the pane sizer.
 * @param {Object} ev The mouseup event that caused us to stop dragging.
 */
CodewalkViewer.prototype.endSizerDrag = function(ev) {
  this.overlay.hide();
  this.updateHeight();
};

/**
 * Toggles the Codewalk description between being shown and hidden.
 * @param {jQuery} target The target that was clicked to trigger this function.
 */
CodewalkViewer.prototype.toggleDescription = function(target) {
  var description = this.context.find('#description');
  description.toggle();
  target.find('span').text(description.is(':hidden') ? 'show' : 'hide');
  this.updateHeight();
};

/**
 * Changes the side of the window on which the code is shown and saves the
 * setting in a cookie.
 * @param {string?} codeSide The side on which the code should be, either
 *     'left' or 'right'.
 */
CodewalkViewer.prototype.changeCodeSide = function(codeSide) {
  var commentSide = codeSide == 'left' ? 'right' : 'left';
  this.context.find('#set-code-' + codeSide).addClass('selected');
  this.context.find('#set-code-' + commentSide).removeClass('selected');
  // Remove previous side class and add new one.
  this.codeColumn.addClass(codeSide).removeClass(commentSide);
  this.commentColumn.addClass(commentSide).removeClass(codeSide);
  this.sizer.css(codeSide, 'auto').css(commentSide, 0);
  this.options.codeSide = codeSide;
};

/**
 * Adds selected class to newly selected comment, removes selected style from
 * previously selected comment, changes drop down options so that the correct
 * file is selected, and updates the code popout link.
 * @param {jQuery} target The target that was clicked to trigger this function.
 */
CodewalkViewer.prototype.changeSelectedComment = function(target) {
  var currentFile = target.find('.comment-link').attr('href');
  if (!currentFile) return;

  if (!(this.lastSelected && this.lastSelected.get(0) === target.get(0))) {
    if (this.lastSelected) this.lastSelected.removeClass('selected');
    target.addClass('selected');
    this.lastSelected = target;
    var targetTop = target.position().top;
    var parentTop = target.parent().position().top;
    if (targetTop + target.height() > parentTop + target.parent().height() ||
        targetTop < parentTop) {
      var delta = targetTop - parentTop;
      target.parent().animate(
          {'scrollTop': target.parent().scrollTop() + delta},
          Math.max(delta / 2, 200), 'swing');
    }
    var fname = currentFile.match(/(?:select=|fileprint=)\/[^&]+/)[0];
    fname = fname.slice(fname.indexOf('=')+2, fname.length);
    this.context.find('#code-selector').val(fname);
    this.context.find('#prev-comment').toggleClass(
        'disabled', !target.prev().length);
    this.context.find('#next-comment').toggleClass(
        'disabled', !target.next().length);
  }

  // Force original file even if user hasn't changed comments since they may
  // have nagivated away from it within the iframe without us knowing.
  this.navigateToCode(currentFile);
};

/**
 * Updates the viewer by changing the height of the comments and code so that
 * they fit within the height of the window.  The function is typically called
 * after the user changes the window size.
 */
CodewalkViewer.prototype.updateHeight = function() {
  var windowHeight = jQuery(window).height() - 5  // GOK
  var areaHeight = windowHeight - this.codeArea.offset().top
  var footerHeight = this.context.find('#footer').outerHeight(true)
  this.commentArea.height(areaHeight - footerHeight - this.context.find('#comment-options').outerHeight(true))
  var codeHeight = areaHeight - footerHeight - 15  // GOK
  this.codeArea.height(codeHeight)
  this.codeDisplay.height(codeHeight - this.codeDisplay.offset().top + this.codeArea.offset().top);
  this.sizer.height(codeHeight);
};

window.initFuncs.push(function() {
  var viewer = new CodewalkViewer(jQuery('#codewalk-main'));
  viewer.selectFirstComment();
  viewer.targetCommentLinksAtBlank();
  viewer.installEventHandlers();
  viewer.updateHeight();
});
