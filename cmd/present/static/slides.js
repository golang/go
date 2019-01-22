// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

var PERMANENT_URL_PREFIX = '/static/';

var SLIDE_CLASSES = ['far-past', 'past', 'current', 'next', 'far-next'];

var PM_TOUCH_SENSITIVITY = 15;

var curSlide;

/* ---------------------------------------------------------------------- */
/* classList polyfill by Eli Grey
 * (http://purl.eligrey.com/github/classList.js/blob/master/classList.js) */

if (typeof document !== 'undefined' && !('classList' in document.createElement('a'))) {

(function (view) {

var
    classListProp = 'classList'
  , protoProp = 'prototype'
  , elemCtrProto = (view.HTMLElement || view.Element)[protoProp]
  , objCtr = Object
    strTrim = String[protoProp].trim || function () {
    return this.replace(/^\s+|\s+$/g, '');
  }
  , arrIndexOf = Array[protoProp].indexOf || function (item) {
    for (var i = 0, len = this.length; i < len; i++) {
      if (i in this && this[i] === item) {
        return i;
      }
    }
    return -1;
  }
  // Vendors: please allow content code to instantiate DOMExceptions
  , DOMEx = function (type, message) {
    this.name = type;
    this.code = DOMException[type];
    this.message = message;
  }
  , checkTokenAndGetIndex = function (classList, token) {
    if (token === '') {
      throw new DOMEx(
          'SYNTAX_ERR'
        , 'An invalid or illegal string was specified'
      );
    }
    if (/\s/.test(token)) {
      throw new DOMEx(
          'INVALID_CHARACTER_ERR'
        , 'String contains an invalid character'
      );
    }
    return arrIndexOf.call(classList, token);
  }
  , ClassList = function (elem) {
    var
        trimmedClasses = strTrim.call(elem.className)
      , classes = trimmedClasses ? trimmedClasses.split(/\s+/) : []
    ;
    for (var i = 0, len = classes.length; i < len; i++) {
      this.push(classes[i]);
    }
    this._updateClassName = function () {
      elem.className = this.toString();
    };
  }
  , classListProto = ClassList[protoProp] = []
  , classListGetter = function () {
    return new ClassList(this);
  }
;
// Most DOMException implementations don't allow calling DOMException's toString()
// on non-DOMExceptions. Error's toString() is sufficient here.
DOMEx[protoProp] = Error[protoProp];
classListProto.item = function (i) {
  return this[i] || null;
};
classListProto.contains = function (token) {
  token += '';
  return checkTokenAndGetIndex(this, token) !== -1;
};
classListProto.add = function (token) {
  token += '';
  if (checkTokenAndGetIndex(this, token) === -1) {
    this.push(token);
    this._updateClassName();
  }
};
classListProto.remove = function (token) {
  token += '';
  var index = checkTokenAndGetIndex(this, token);
  if (index !== -1) {
    this.splice(index, 1);
    this._updateClassName();
  }
};
classListProto.toggle = function (token) {
  token += '';
  if (checkTokenAndGetIndex(this, token) === -1) {
    this.add(token);
  } else {
    this.remove(token);
  }
};
classListProto.toString = function () {
  return this.join(' ');
};

if (objCtr.defineProperty) {
  var classListPropDesc = {
      get: classListGetter
    , enumerable: true
    , configurable: true
  };
  try {
    objCtr.defineProperty(elemCtrProto, classListProp, classListPropDesc);
  } catch (ex) { // IE 8 doesn't support enumerable:true
    if (ex.number === -0x7FF5EC54) {
      classListPropDesc.enumerable = false;
      objCtr.defineProperty(elemCtrProto, classListProp, classListPropDesc);
    }
  }
} else if (objCtr[protoProp].__defineGetter__) {
  elemCtrProto.__defineGetter__(classListProp, classListGetter);
}

}(self));

}
/* ---------------------------------------------------------------------- */

/* Slide movement */

function hideHelpText() {
  document.getElementById('help').style.display = 'none';
};

function getSlideEl(no) {
  if ((no < 0) || (no >= slideEls.length)) {
    return null;
  } else {
    return slideEls[no];
  }
};

function updateSlideClass(slideNo, className) {
  var el = getSlideEl(slideNo);

  if (!el) {
    return;
  }

  if (className) {
    el.classList.add(className);
  }

  for (var i in SLIDE_CLASSES) {
    if (className != SLIDE_CLASSES[i]) {
      el.classList.remove(SLIDE_CLASSES[i]);
    }
  }
};

function updateSlides() {
  if (window.trackPageview) window.trackPageview();

  for (var i = 0; i < slideEls.length; i++) {
    switch (i) {
      case curSlide - 2:
        updateSlideClass(i, 'far-past');
        break;
      case curSlide - 1:
        updateSlideClass(i, 'past');
        break;
      case curSlide:
        updateSlideClass(i, 'current');
        break;
      case curSlide + 1:
        updateSlideClass(i, 'next');
        break;
      case curSlide + 2:
        updateSlideClass(i, 'far-next');
        break;
      default:
        updateSlideClass(i);
        break;
    }
  }

  triggerLeaveEvent(curSlide - 1);
  triggerEnterEvent(curSlide);

  window.setTimeout(function() {
    // Hide after the slide
    disableSlideFrames(curSlide - 2);
  }, 301);

  enableSlideFrames(curSlide - 1);
  enableSlideFrames(curSlide + 2);

  updateHash();
};

function prevSlide() {
  hideHelpText();
  if (curSlide > 0) {
    curSlide--;

    updateSlides();
  }

  if (notesEnabled) localStorage.setItem('destSlide', curSlide);
};

function nextSlide() {
  hideHelpText();
  if (curSlide < slideEls.length - 1) {
    curSlide++;

    updateSlides();
  }

  if (notesEnabled) localStorage.setItem('destSlide', curSlide);
};

/* Slide events */

function triggerEnterEvent(no) {
  var el = getSlideEl(no);
  if (!el) {
    return;
  }

  var onEnter = el.getAttribute('onslideenter');
  if (onEnter) {
    new Function(onEnter).call(el);
  }

  var evt = document.createEvent('Event');
  evt.initEvent('slideenter', true, true);
  evt.slideNumber = no + 1; // Make it readable

  el.dispatchEvent(evt);
};

function triggerLeaveEvent(no) {
  var el = getSlideEl(no);
  if (!el) {
    return;
  }

  var onLeave = el.getAttribute('onslideleave');
  if (onLeave) {
    new Function(onLeave).call(el);
  }

  var evt = document.createEvent('Event');
  evt.initEvent('slideleave', true, true);
  evt.slideNumber = no + 1; // Make it readable

  el.dispatchEvent(evt);
};

/* Touch events */

function handleTouchStart(event) {
  if (event.touches.length == 1) {
    touchDX = 0;
    touchDY = 0;

    touchStartX = event.touches[0].pageX;
    touchStartY = event.touches[0].pageY;

    document.body.addEventListener('touchmove', handleTouchMove, true);
    document.body.addEventListener('touchend', handleTouchEnd, true);
  }
};

function handleTouchMove(event) {
  if (event.touches.length > 1) {
    cancelTouch();
  } else {
    touchDX = event.touches[0].pageX - touchStartX;
    touchDY = event.touches[0].pageY - touchStartY;
    event.preventDefault();
  }
};

function handleTouchEnd(event) {
  var dx = Math.abs(touchDX);
  var dy = Math.abs(touchDY);

  if ((dx > PM_TOUCH_SENSITIVITY) && (dy < (dx * 2 / 3))) {
    if (touchDX > 0) {
      prevSlide();
    } else {
      nextSlide();
    }
  }

  cancelTouch();
};

function cancelTouch() {
  document.body.removeEventListener('touchmove', handleTouchMove, true);
  document.body.removeEventListener('touchend', handleTouchEnd, true);
};

/* Preloading frames */

function disableSlideFrames(no) {
  var el = getSlideEl(no);
  if (!el) {
    return;
  }

  var frames = el.getElementsByTagName('iframe');
  for (var i = 0, frame; frame = frames[i]; i++) {
    disableFrame(frame);
  }
};

function enableSlideFrames(no) {
  var el = getSlideEl(no);
  if (!el) {
    return;
  }

  var frames = el.getElementsByTagName('iframe');
  for (var i = 0, frame; frame = frames[i]; i++) {
    enableFrame(frame);
  }
};

function disableFrame(frame) {
  frame.src = 'about:blank';
};

function enableFrame(frame) {
  var src = frame._src;

  if (frame.src != src && src != 'about:blank') {
    frame.src = src;
  }
};

function setupFrames() {
  var frames = document.querySelectorAll('iframe');
  for (var i = 0, frame; frame = frames[i]; i++) {
    frame._src = frame.src;
    disableFrame(frame);
  }

  enableSlideFrames(curSlide);
  enableSlideFrames(curSlide + 1);
  enableSlideFrames(curSlide + 2);
};

function setupInteraction() {
  /* Clicking and tapping */

  var el = document.createElement('div');
  el.className = 'slide-area';
  el.id = 'prev-slide-area';
  el.addEventListener('click', prevSlide, false);
  document.querySelector('section.slides').appendChild(el);

  var el = document.createElement('div');
  el.className = 'slide-area';
  el.id = 'next-slide-area';
  el.addEventListener('click', nextSlide, false);
  document.querySelector('section.slides').appendChild(el);

  /* Swiping */

  document.body.addEventListener('touchstart', handleTouchStart, false);
}

/* Hash functions */

function getCurSlideFromHash() {
  var slideNo = parseInt(location.hash.substr(1));

  if (slideNo) {
    curSlide = slideNo - 1;
  } else {
    curSlide = 0;
  }
};

function updateHash() {
  location.replace('#' + (curSlide + 1));
};

/* Event listeners */

function handleBodyKeyDown(event) {
  // If we're in a code element, only handle pgup/down.
  var inCode = event.target.classList.contains('code');

  switch (event.keyCode) {
    case 78: // 'N' opens presenter notes window
      if (!inCode && notesEnabled) toggleNotesWindow();
      break;
    case 72: // 'H' hides the help text
    case 27: // escape key
      if (!inCode) hideHelpText();
      break;

    case 39: // right arrow
    case 13: // Enter
    case 32: // space
      if (inCode) break;
    case 34: // PgDn
      nextSlide();
      event.preventDefault();
      break;

    case 37: // left arrow
    case 8: // Backspace
      if (inCode) break;
    case 33: // PgUp
      prevSlide();
      event.preventDefault();
      break;

    case 40: // down arrow
      if (inCode) break;
      nextSlide();
      event.preventDefault();
      break;

    case 38: // up arrow
      if (inCode) break;
      prevSlide();
      event.preventDefault();
      break;
  }
};

function scaleSmallViewports() {
  var el = document.querySelector('section.slides');
  var transform = '';
  var sWidthPx = 1250;
  var sHeightPx = 750;
  var sAspectRatio = sWidthPx / sHeightPx;
  var wAspectRatio = window.innerWidth / window.innerHeight;

  if (wAspectRatio <= sAspectRatio && window.innerWidth < sWidthPx) {
    transform = 'scale(' + window.innerWidth / sWidthPx + ')';
  } else if (window.innerHeight < sHeightPx) {
    transform = 'scale(' + window.innerHeight / sHeightPx + ')';
  }
  el.style.transform = transform;
}

function addEventListeners() {
  document.addEventListener('keydown', handleBodyKeyDown, false);
  var resizeTimeout;
  window.addEventListener('resize', function() {
    // throttle resize events
    window.clearTimeout(resizeTimeout);
    resizeTimeout = window.setTimeout(function() {
      resizeTimeout = null;
      scaleSmallViewports();
    }, 50);
  });

  // Force reset transform property of section.slides when printing page.
  // Use both onbeforeprint and matchMedia for compatibility with different browsers.
  var beforePrint = function() {
    var el = document.querySelector('section.slides');
    el.style.transform = '';
  };
  window.onbeforeprint = beforePrint;
  if (window.matchMedia) {
    var mediaQueryList = window.matchMedia('print');
    mediaQueryList.addListener(function(mql) {
      if (mql.matches) beforePrint();
    });
  }
}

/* Initialization */

function addFontStyle() {
  var el = document.createElement('link');
  el.rel = 'stylesheet';
  el.type = 'text/css';
  el.href = '//fonts.googleapis.com/css?family=' +
            'Open+Sans:regular,semibold,italic,italicsemibold|Droid+Sans+Mono';

  document.body.appendChild(el);
};

function addGeneralStyle() {
  var el = document.createElement('link');
  el.rel = 'stylesheet';
  el.type = 'text/css';
  el.href = PERMANENT_URL_PREFIX + 'styles.css';
  document.body.appendChild(el);

  var el = document.createElement('meta');
  el.name = 'viewport';
  el.content = 'width=device-width,height=device-height,initial-scale=1';
  document.querySelector('head').appendChild(el);

  var el = document.createElement('meta');
  el.name = 'apple-mobile-web-app-capable';
  el.content = 'yes';
  document.querySelector('head').appendChild(el);

  scaleSmallViewports();
};

function handleDomLoaded() {
  slideEls = document.querySelectorAll('section.slides > article');

  setupFrames();

  addFontStyle();
  addGeneralStyle();
  addEventListeners();

  updateSlides();

  setupInteraction();

  if (window.location.hostname == 'localhost' || window.location.hostname == '127.0.0.1' || window.location.hostname == '::1') {
    hideHelpText();
  }

  document.body.classList.add('loaded');

  setupNotesSync();
};

function initialize() {
  getCurSlideFromHash();

  if (window['_DEBUG']) {
    PERMANENT_URL_PREFIX = '../';
  }

  if (window['_DCL']) {
    handleDomLoaded();
  } else {
    document.addEventListener('DOMContentLoaded', handleDomLoaded, false);
  }
}

// If ?debug exists then load the script relative instead of absolute
if (!window['_DEBUG'] && document.location.href.indexOf('?debug') !== -1) {
  document.addEventListener('DOMContentLoaded', function() {
    // Avoid missing the DomContentLoaded event
    window['_DCL'] = true
  }, false);

  window['_DEBUG'] = true;
  var script = document.createElement('script');
  script.type = 'text/javascript';
  script.src = '../slides.js';
  var s = document.getElementsByTagName('script')[0];
  s.parentNode.insertBefore(script, s);

  // Remove this script
  s.parentNode.removeChild(s);
} else {
  initialize();
}

/* Synchronize windows when notes are enabled */

function setupNotesSync() {
  if (!notesEnabled) return;

  function setupPlayResizeSync() {
    var out = document.getElementsByClassName('output');
    for (var i = 0; i < out.length; i++) {
      $(out[i]).bind('resize', function(event) {
        if ($(event.target).hasClass('ui-resizable')) {
          localStorage.setItem('play-index', i);
          localStorage.setItem('output-style', out[i].style.cssText);
        }
      })
    }
  };
  function setupPlayCodeSync() {
    var play = document.querySelectorAll('div.playground');
    for (var i = 0; i < play.length; i++) {
      play[i].addEventListener('input', inputHandler, false);

      function inputHandler(e) {
        localStorage.setItem('play-index', i);
        localStorage.setItem('play-code', e.target.innerHTML);
      }
    }
  };

  setupPlayCodeSync();
  setupPlayResizeSync();
  localStorage.setItem('destSlide', curSlide);
  window.addEventListener('storage', updateOtherWindow, false);
}

// An update to local storage is caught only by the other window
// The triggering window does not handle any sync actions
function updateOtherWindow(e) {
  // Ignore remove storage events which are not meant to update the other window
  var isRemoveStorageEvent = !e.newValue;
  if (isRemoveStorageEvent) return;

  var destSlide = localStorage.getItem('destSlide');
  while (destSlide > curSlide) {
    nextSlide();
  }
  while (destSlide < curSlide) {
    prevSlide();
  }

  updatePlay(e);
  updateNotes();
}
