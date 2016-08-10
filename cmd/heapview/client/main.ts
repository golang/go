// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/**
 * An enum of types of actions that might be requested
 * by the app.
 */
enum Action {
  TOGGLE_SIDEBAR,  // Toggle the sidebar.
  NAVIGATE_ABOUT,  // Go to the about page.
}

const TITLE = 'Go Heap Viewer';

/**
 * A type of event that signals to the AppElement controller
 * that something shoud be done. For the most part, the structure
 * of the app will be that elements' state will mostly be controlled
 * by parent elements. Elements will issue actions that the AppElement
 * will handle, and the app will be re-rendered down the DOM
 * hierarchy.
 */
class ActionEvent extends Event {
  static readonly EVENT_TYPE = 'action-event'
  constructor(public readonly action: Action) { super(ActionEvent.EVENT_TYPE); }
}

/**
 * A hamburger menu element. Triggers a TOGGLE_SIDE action to toggle the
 * sidebar.
 */
export class HamburgerElement extends HTMLElement {
  static readonly NAME = 'heap-hamburger';

  createdCallback() {
    this.appendChild(document.createTextNode('☰'));
    this.onclick =
        () => { this.dispatchEvent(new ActionEvent(Action.TOGGLE_SIDEBAR)) };
  }
}
document.registerElement(HamburgerElement.NAME, HamburgerElement);

/**
 * A heading for the page with a hamburger menu and a title.
 */
export class HeadingElement extends HTMLElement {
  static readonly NAME = 'heap-heading';

  createdCallback() {
    this.style.display = 'block';
    this.style.backgroundColor = '#2196F3';
    this.style.webkitUserSelect = 'none';
    this.style.cursor = 'default';
    this.style.color = '#FFFFFF';
    this.style.padding = '10px';

    const div = document.createElement('div');
    div.style.margin = '0px';
    div.style.fontSize = '2em';
    div.appendChild(document.createElement(HamburgerElement.NAME));
    div.appendChild(document.createTextNode(' ' + TITLE));
    this.appendChild(div);
  }
}
document.registerElement(HeadingElement.NAME, HeadingElement);

/**
 * A sidebar that has navigation for the app.
 */
export class SidebarElement extends HTMLElement {
  static readonly NAME = 'heap-sidebar';

  createdCallback() {
    this.style.display = 'none';
    this.style.backgroundColor = '#9E9E9E';
    this.style.width = '15em';

    const aboutButton = document.createElement('button');
    aboutButton.innerText = 'about';
    aboutButton.onclick =
        () => { this.dispatchEvent(new ActionEvent(Action.NAVIGATE_ABOUT)) };
    this.appendChild(aboutButton);
  }

  toggle() {
    this.style.display = this.style.display === 'none' ? 'block' : 'none';
  }
}
document.registerElement(SidebarElement.NAME, SidebarElement);

/**
 * A Container for the main content in the app.
 * TODO(matloob): Implement main content.
 */
export class MainContentElement extends HTMLElement {
  static readonly NAME = 'heap-container';

  attachedCallback() {
    this.style.backgroundColor = '#E0E0E0';
    this.style.height = '100%';
    this.style.flex = '1';
  }
}
document.registerElement(MainContentElement.NAME, MainContentElement);

/**
 * A container and controller for the whole app.
 * Contains the heading, side drawer and main panel.
 */
class AppElement extends HTMLElement {
  static readonly NAME = 'heap-app';
  private sidebar: SidebarElement;
  private mainContent: MainContentElement;

  attachedCallback() {
    document.title = TITLE;

    this.addEventListener(
        ActionEvent.EVENT_TYPE, e => this.handleAction(e as ActionEvent),
        /* capture */ true);

    this.render();
  }

  render() {
    this.style.display = 'block';
    this.style.height = '100vh';
    this.style.width = '100vw';
    this.appendChild(document.createElement(HeadingElement.NAME));

    const bodyDiv = document.createElement('div');
    bodyDiv.style.height = '100%';
    bodyDiv.style.display = 'flex';
    this.sidebar =
        document.createElement(SidebarElement.NAME) as SidebarElement;
    bodyDiv.appendChild(this.sidebar);
    this.mainContent =
        document.createElement(MainContentElement.NAME) as MainContentElement;
    bodyDiv.appendChild(this.mainContent);
    this.appendChild(bodyDiv);

    this.renderRoute();
  }

  renderRoute() {
    this.mainContent.innerHTML = ''
    switch (window.location.pathname) {
      case '/about':
        this.mainContent.appendChild(
            document.createElement(AboutPageElement.NAME));
        break;
    }
  }

  handleAction(event: ActionEvent) {
    switch (event.action) {
      case Action.TOGGLE_SIDEBAR:
        this.sidebar.toggle();
        break;
      case Action.NAVIGATE_ABOUT:
        window.history.pushState({}, '', '/about');
        this.renderRoute();
        break;
    }
  }
}
document.registerElement(AppElement.NAME, AppElement);

/**
 * An about page.
 */
class AboutPageElement extends HTMLElement {
  static readonly NAME = 'heap-about';

  createdCallback() { this.textContent = TITLE; }
}
document.registerElement(AboutPageElement.NAME, AboutPageElement);

/**
 * Resets body's margin and padding, and sets font.
 */
function clearStyle(document: Document) {
  const styleElement = document.createElement('style') as HTMLStyleElement;
  document.head.appendChild(styleElement);
  const styleSheet = styleElement.sheet as CSSStyleSheet;
  styleSheet.insertRule(
      '* {font-family: Roboto,Helvetica; box-sizing: border-box}', 0);
  styleSheet.insertRule('body {margin: 0px; padding:0px}', 0);
}

export function main() {
  clearStyle(document);
  document.body.appendChild(document.createElement(AppElement.NAME));
}
